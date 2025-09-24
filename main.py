# sensory_coach_with_audio.py
import sys
import uuid
import time
import threading
import queue
import numpy as np
from fer import FER
from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QSlider, QTextEdit, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QImage
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import pygame

# --- NEW audio libs ---
import sounddevice as sd
import librosa
import collections
from threading import Lock

# ---------- CONFIG ----------
MIC_DEVICE_INDEX = 1   # adjust this if your mic is at a different index
device_info = sd.query_devices(MIC_DEVICE_INDEX, "input")
MIC_CHANNELS = device_info["max_input_channels"] if device_info["max_input_channels"] > 0 else 1

session_id = str(uuid.uuid4())
session_start = None
session_log = []
metrics = {}
cap = None
running = False

detector = FER(mtcnn=False)
pygame.mixer.init()
MUSIC_FILE = "soothing_music.mp3"

thresholds = {"calm": 30, "mild": 60}
messages = {
    "Calm": "Great job staying relaxed!",
    "Mild Stress": "You're doing fine. Take a deep breath.",
    "Overload Risk": "Pause for a moment, you've got this."
}

VIDEO_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4

# ---------- UTIL ----------
def get_stress_score(emotion, score):
    if not emotion:
        return 0.0
    if emotion in ["happy", "neutral"]:
        return -15 * score
    elif emotion == "surprise":
        return 10 * score
    elif emotion in ["fear", "angry", "sad", "disgust"]:
        return 20 * score
    return 0

def start_music(volume=0.3):
    try:
        pygame.mixer.music.load(MUSIC_FILE)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print("Could not start music:", e)

def set_music_volume(volume):
    try:
        pygame.mixer.music.set_volume(volume)
    except Exception:
        pass

def stop_music():
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

# =========================
# Audio listener
# =========================
class AudioListener:
    def __init__(self, device=None, samplerate=16000, block_duration=0.6, channels=1):
        self.samplerate = samplerate
        self.block_duration = block_duration
        self.blocksize = int(samplerate * block_duration)
        self.device = device
        self.channels = channels
        self._q = queue.Queue(maxsize=40)
        self._stream = None
        self._thread = None
        self.running = False
        self.lock = Lock()
        self.latest_emotion = None
        self.latest_score = 0.0
        self.history = collections.deque(maxlen=6)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            pass
        arr = indata.copy()
        if arr.ndim > 1:  # average to mono
            arr = np.mean(arr, axis=1)
        try:
            self._q.put_nowait(arr)
        except queue.Full:
            pass

    def _processing_loop(self):
        while self.running:
            try:
                block = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                emotion, conf = self._infer_emotion_from_audio(block)
            except Exception as e:
                print("Audio processing error:", e)
                continue

            with self.lock:
                self.history.append((emotion, conf))
                votes = {}
                for emo, c in self.history:
                    votes.setdefault(emo, 0.0)
                    votes[emo] += c
                best = max(votes.items(), key=lambda x: x[1])
                self.latest_emotion = best[0]
                self.latest_score = float(min(1.0, votes[best[0]] / (len(self.history) + 1e-6)))
            time.sleep(0.01)

    def _infer_emotion_from_audio(self, audio_block):
        audio = audio_block.astype(np.float32).flatten()
        if audio.size == 0:
            return ("neutral", 0.0)
        peak = np.max(np.abs(audio)) + 1e-9
        audio = audio / peak

        try:
            rms = float(np.sqrt(np.mean(audio**2)))
        except Exception:
            rms = 0.0
        try:
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.samplerate)))
        except Exception:
            centroid = 0.0
        try:
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        except Exception:
            zcr = 0.0
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=self.samplerate, n_mfcc=13)
            mfcc0 = float(np.mean(mfcc[0]))
        except Exception:
            mfcc0 = 0.0

        act_norm = min(1.0, rms * 10.0)
        bright_norm = min(1.0, centroid / (self.samplerate/4 + 1e-9))
        zcr_norm = min(1.0, zcr * 5.0)

        angry_score = max(0.0, act_norm * 0.6 + bright_norm * 0.3 + zcr_norm * 0.1)
        sad_score = max(0.0, (1 - act_norm) * 0.6 + (1 - bright_norm) * 0.3)
        neutral_score = max(0.0, 1 - abs(act_norm - 0.4) - abs(bright_norm - 0.35))
        happy_score = max(0.0, 0.5*act_norm + 0.5*(1 - abs(mfcc0)/50.0))
        surprise_score = max(0.0, 0.8*act_norm * bright_norm)

        score_map = {
            "angry": angry_score,
            "sad": sad_score,
            "neutral": neutral_score,
            "happy": happy_score,
            "surprise": surprise_score
        }
        label, raw_score = max(score_map.items(), key=lambda x: x[1])
        conf = float(min(1.0, raw_score))
        if conf < 0.12:
            label = "neutral"
            conf = max(conf, 0.2)
        return (label, conf)

    def start(self):
        if self.running:
            return
        self.running = True
        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                device=self.device,
                channels=self.channels,
                blocksize=self.blocksize,
                callback=self._audio_callback
            )
            self._stream.start()
            print(f"Audio stream started on device {self.device} with {self.channels} channel(s)")
        except Exception as e:
            print("Audio input stream error:", e)
            self.running = False
            return

        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        with self._q.mutex:
            self._q.queue.clear()

# global audio listener
audio_listener = AudioListener(device=MIC_DEVICE_INDEX, samplerate=16000, block_duration=0.6, channels=MIC_CHANNELS)

# =========================
# GUI (mostly as original) with audio integration improvements
# =========================

def combine_scores(video_emotion, video_score, audio_emotion, audio_score):
    """
    Combine video + audio contributions into a single stress delta.
    video_score and audio_score are 0..1 confidences as returned from FER / audio heuristics
    """
    v_contrib = get_stress_score(video_emotion, video_score) if video_emotion else 0.0
    a_contrib = get_stress_score(audio_emotion, audio_score) if audio_emotion else 0.0
    combined = VIDEO_WEIGHT * v_contrib + AUDIO_WEIGHT * a_contrib
    return combined

class SensoryCoachApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SensoryCoach")
        self.setGeometry(100, 100, 1280, 720)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.start_page = StartPage(self)
        self.monitor_page = MonitorPage(self)
        self.feedback_page = CaregiverInsightsPage(self)
        self.summary_page = SummaryPage(self)
        self.graphs_page = GraphsPage(self)

        for page in [self.start_page, self.monitor_page, self.feedback_page, self.summary_page, self.graphs_page]:
            self.stack.addWidget(page)

        self.show_start()

    def show_start(self):
        self.stack.setCurrentWidget(self.start_page)

    def show_monitor(self):
        self.stack.setCurrentWidget(self.monitor_page)
        self.start_session()

    def show_feedback(self):
        self.stack.setCurrentWidget(self.feedback_page)

    def show_summary(self):
        self.stack.setCurrentWidget(self.summary_page)
        self.summary_page.display_summary()

    def show_graphs(self):
        self.stack.setCurrentWidget(self.graphs_page)
        self.graphs_page.display_graphs()

    def go_back(self, current_page):
        if current_page == self.monitor_page:
            self.show_start()
        elif current_page == self.feedback_page:
            self.show_monitor()
        elif current_page == self.summary_page:
            self.show_feedback()
        elif current_page == self.graphs_page:
            self.show_summary()

    def start_session(self):
        global cap, running, session_start, session_log, audio_listener
        session_log = []
        session_start = time.time()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        running = True
        start_music(0.3)
        # START audio listener
        try:
            audio_listener.start()
        except Exception as e:
            print("Could not start audio listener:", e)
        self.monitor_page.start_timer()

    def finish_session(self):
        global running, metrics, session_log, cap, session_start, audio_listener
        running = False
        if cap:
            cap.release()
        stop_music()
        # stop audio
        try:
            audio_listener.stop()
        except Exception:
            pass

        session_end = time.time()
        duration = session_end - session_start
        stress_values = [e["stress_score"] for e in session_log]
        states = [e["state"] for e in session_log]
        emotions = [e["emotion"] for e in session_log if e["emotion"]]
        volumes = [e["volume"] for e in session_log]

        avg_stress = float(np.mean(stress_values)) if stress_values else 0
        std_stress = float(np.std(stress_values)) if stress_values else 0
        overload_events = states.count("Overload Risk")
        time_in_states = {
            "Calm": states.count("Calm") / max(1, len(states)) * 100,
            "Mild Stress": states.count("Mild Stress") / max(1, len(states)) * 100,
            "Overload Risk": states.count("Overload Risk") / max(1, len(states)) * 100
        }
        emotion_counts = {emo: emotions.count(emo) for emo in set(emotions)}
        emotional_balance_index = (emotion_counts.get("happy",0)+emotion_counts.get("neutral",0))/max(1,len(emotions))

        recovery_intervals = []
        current_overload = False
        start_idx = 0
        for idx, s in enumerate(states):
            if s == "Overload Risk" and not current_overload:
                current_overload = True
                start_idx = idx
            elif s != "Overload Risk" and current_overload:
                recovery_intervals.append(idx - start_idx)
                current_overload = False
        recovery_score = 1 - (sum(recovery_intervals)/max(1,len(states)))

        engagement_duration = duration
        avg_volume = np.mean(volumes) if volumes else 0

        metrics.update({
            "session_id": session_id,
            "session_duration_sec": duration,
            "avg_stress_score": avg_stress,
            "stress_stability": std_stress,
            "time_in_states_percent": time_in_states,
            "overload_events": overload_events,
            "emotion_distribution": emotion_counts,
            "emotional_balance_index": emotional_balance_index,
            "recovery_score": recovery_score,
            "engagement_duration_sec": engagement_duration,
            "avg_music_volume": avg_volume
        })

        if "caregiver_ratings" in metrics:
            cr = metrics["caregiver_ratings"]
            metrics["caregiver_ratings"] = {k.replace("_"," "): v for k,v in cr.items()}

        session_entry = {
            "session_id": session_id,
            "session_duration_sec": duration,
            "avg_stress_score": avg_stress,
            "stress_stability": std_stress,
            "time_in_states_percent": time_in_states,
            "overload_events": overload_events,
            "emotion_distribution": emotion_counts,
            "emotional_balance_index": emotional_balance_index,
            "recovery_score": recovery_score,
            "engagement_duration_sec": engagement_duration,
            "avg_music_volume": avg_volume,
            "caregiver_ratings": metrics.get("caregiver_ratings", {})
        }

        import json
        try:
            with open("session_log.json", "r") as f:
                all_logs = json.load(f)
        except FileNotFoundError:
            all_logs = [{"sessions": []}]

        all_logs[0]["sessions"].append(session_entry)

        with open("session_log.json", "w") as f:
            json.dump(all_logs, f, indent=2)

        try:
            with open("session_metrics.json", "r") as f:
                all_metrics = json.load(f)
        except FileNotFoundError:
            all_metrics = {"metrics": []}

        all_metrics["metrics"].append(metrics)

        with open("session_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        self.show_feedback()

class StartPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        label = QLabel("=== Welcome to SensoryCoach ===")
        label.setStyleSheet("font-size: 28px; font-weight: bold; color: #FFD700;")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        desc = QLabel("Sit comfortably, breathe deeply.\nLook at the camera when ready.")
        desc.setStyleSheet("font-size: 18px; color: #DDDDDD;")
        desc.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        start_btn = QPushButton("Start Session")
        start_btn.setFixedWidth(220)
        start_btn.setStyleSheet("font-size:16px; padding:10px; background-color:#00CED1; color:white; border-radius:8px;")
        start_btn.clicked.connect(main.show_monitor)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(220)
        close_btn.setStyleSheet("font-size:16px; padding:10px; background-color:#FF6347; color:white; border-radius:8px;")
        close_btn.clicked.connect(sys.exit)

        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(start_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

class MonitorPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        layout = QVBoxLayout()
        layout.setContentsMargins(10,10,10,10)
        layout.setSpacing(5)

        self.label = QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.label, stretch=1)

        btn_layout = QHBoxLayout()
        finish_btn = QPushButton("Next")
        finish_btn.clicked.connect(main.finish_session)
        finish_btn.setStyleSheet("padding:10px; background-color:#00CED1; color:white; border-radius:6px;")
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: main.go_back(self))
        back_btn.setStyleSheet("padding:10px; background-color:#FFA500; color:white; border-radius:6px;")
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(sys.exit)
        close_btn.setStyleSheet("padding:10px; background-color:#FF6347; color:white; border-radius:6px;")
        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(finish_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.frame_counter = 0
        self.dominant_emotion, self.emo_score = None, 0
        self.state = "Calm"
        self.color = (0, 255, 255)
        self.new_volume = 0.3
        self.stress_score = 0

        # Keep last known non-zero scores for reliable display
        self.last_video_score = 0.0
        self.last_audio_score = 0.0
        self.last_video_emotion = None
        self.last_audio_emotion = None

    def start_timer(self):
        self.timer.start(30)

    def update_frame(self):
        global running, session_log, cap, audio_listener
        if not running:
            self.timer.stop()
            return
        ret, frame = cap.read()
        if not ret:
            return

        h_label = self.label.height()
        w_label = self.label.width()
        # protect against tiny widget sizes
        if h_label < 50 or w_label < 50:
            h_label, w_label = 480, 640
        frame = cv2.resize(frame, (w_label, int(h_label*0.9)))

        self.frame_counter += 1

        video_emotion, video_score = (None, 0.0)
        audio_emotion, audio_score = (None, 0.0)

        # Video: sample every N frames for performance
        if self.frame_counter % 10 == 0:
            small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            try:
                emotions = detector.detect_emotions(small_frame)
                if emotions:
                    video_emotion, video_score = detector.top_emotion(small_frame)
                    video_score = float(np.clip(video_score, 0.0, 1.0))
                else:
                    video_emotion, video_score = (None, 0.0)
            except Exception as e:
                video_emotion, video_score = (None, 0.0)
                # print("FER error:", e)

            # Read latest audio emotion (thread-safe)
            with audio_listener.lock:
                audio_emotion = audio_listener.latest_emotion
                audio_score = float(audio_listener.latest_score)

            # keep last valid values if current are None/0
            if video_emotion:
                self.last_video_emotion = video_emotion
            if video_score and video_score > 0:
                self.last_video_score = video_score
            if audio_emotion:
                self.last_audio_emotion = audio_emotion
            if audio_score and audio_score > 0:
                self.last_audio_score = audio_score

            # combine using last-known valid values for display and for stress delta
            used_video_emotion = video_emotion or self.last_video_emotion
            used_video_score = video_score or self.last_video_score
            used_audio_emotion = audio_emotion or self.last_audio_emotion
            used_audio_score = audio_score or self.last_audio_score

            combined_delta = combine_scores(used_video_emotion, used_video_score, used_audio_emotion, used_audio_score)
            # Apply an extra scaling to make stress changes visible but smooth
            # (you may tune or remove this scale)
            self.stress_score += combined_delta

            # keep internal storage of last detected emotions for display
            self.dominant_emotion = used_video_emotion or used_audio_emotion
            self.emo_score = max(used_video_score, used_audio_score)

        # clamp stress to 0..100
        self.stress_score = max(0, min(100, self.stress_score))

        # map stress to states (same as before)
        if self.stress_score < thresholds["calm"]:
            self.state = "Calm"
            self.color = (0, 255, 255)
            self.new_volume = 0.3
        elif self.stress_score < thresholds["mild"]:
            self.state = "Mild Stress"
            self.color = (0, 255, 0)
            self.new_volume = min(0.7, 0.3 + self.emo_score)
        else:
            self.state = "Overload Risk"
            self.color = (0, 0, 255)
            self.new_volume = min(1.0, 0.5 + self.emo_score)

        set_music_volume(self.new_volume)

        session_log.append({
            "session_id": session_id,
            "time": time.time(),
            "emotion": self.dominant_emotion,
            "score": float(self.emo_score),
            "audio_emotion": audio_emotion,
            "audio_score": float(audio_score),
            "stress_score": float(self.stress_score),
            "state": self.state,
            "volume": float(self.new_volume)
        })

        # Drawing UI overlay on frame (same as original)
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0,0), (w,100), (20,20,20), -1)
        cv2.putText(frame, "SensoryCoach - Stress Monitor", (20,40),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 2)
        cv2.rectangle(frame, (20,60), (200,90), self.color, -1)
        cv2.putText(frame, self.state, (30,85), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,0,0), 2)

        bar_x, bar_y, bar_w, bar_h = 220, 70, 300, 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100,100,100), 2)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * (self.stress_score / 100)), bar_y + bar_h), self.color, -1)
        cv2.putText(frame, f"Stress: {int(self.stress_score)}", (bar_x + bar_w + 20, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        vol_x, vol_y, vol_w, vol_h = 560, 70, 150, 20
        cv2.rectangle(frame, (vol_x, vol_y), (vol_x + vol_w, vol_y + vol_h), (100,100,100), 2)
        cv2.rectangle(frame, (vol_x, vol_y),
                      (vol_x + int(vol_w * self.new_volume), vol_y + vol_h), (0,150,255), -1)
        cv2.putText(frame, f"Volume: {self.new_volume:.2f}", (vol_x + vol_w + 20, vol_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.rectangle(frame, (0, h-80), (w,h), (20,20,20), -1)
        # show both detected emotions line - use last-known values so 0.00 doesn't stick
        disp_video_emotion = (video_emotion or self.last_video_emotion or "-")
        disp_video_score = (video_score or self.last_video_score or 0.0)
        disp_audio_emotion = (audio_emotion or self.last_audio_emotion or "-")
        disp_audio_score = (audio_score or self.last_audio_score or 0.0)
        disp_line = f"Video: {disp_video_emotion} ({disp_video_score:.2f})  |  Audio: {disp_audio_emotion} ({disp_audio_score:.2f})"
        cv2.putText(frame, disp_line, (30, h-40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, messages[self.state], (30, h-15), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qt_img).scaled(w_label, h_label, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.label.setPixmap(pix)

class CaregiverInsightsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QVBoxLayout()
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(12)

        title = QLabel("💛 Caregiver Feedback")
        title.setStyleSheet("font-size:24px; font-weight:bold; color:yellow;")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        desc = QLabel("Please provide your observations about the participant's session.\nUse sliders for detailed feedback.")
        desc.setStyleSheet("font-size:16px; color:#EEEEEE;")
        desc.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        self.sliders = {}
        feedback_questions = [
            ("Calmness Level", "0 = Not Calm, 10 = Very Calm", "#00CED1"),
            ("Attention Level", "0 = Distracted, 10 = Fully Attentive", "#FF8C00"),
            ("Willingness to Engage", "0 = Unwilling, 10 = Fully Engaged", "#32CD32"),
            ("Emotional Response", "0 = Negative, 10 = Positive", "#FF69B4"),
            ("Engagement Duration", "0 = Short, 10 = Long", "#8A2BE2")
        ]

        for q, tip, color in feedback_questions:
            lbl = QLabel(q)
            lbl.setStyleSheet(f"font-size:16px; color:{color}; font-weight:bold;")
            slider = QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(10)
            slider.setValue(5)
            slider.setTickInterval(1)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setToolTip(tip)
            layout.addWidget(lbl)
            layout.addWidget(slider)
            self.sliders[q.lower().replace(" ","_")] = slider

        btn_layout = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: main.go_back(self))
        back_btn.setStyleSheet("padding:10px; background-color:#FFA500; color:white; border-radius:8px;")
        submit_btn = QPushButton("Submit and Next → Metrics")
        submit_btn.setStyleSheet("padding:10px; background-color:#4B0082; color:white; font-weight:bold; border-radius:8px;")
        submit_btn.clicked.connect(self.save_feedback)
        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(submit_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def save_feedback(self):
        global metrics
        caregiver_ratings = {k: v.value() for k,v in self.sliders.items()}
        metrics["caregiver_ratings"] = caregiver_ratings
        self.main.show_summary()

class SummaryPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20,20,20,20)

        self.title = QLabel("✨ Session Metrics ✨")
        self.title.setStyleSheet("font-size:28px; font-weight:bold; color:#00CED1;")
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        self.text_box.setStyleSheet("font-size:16px; color:#FFFFFF; background-color:#2F2F2F; border-radius:8px; padding:10px;")
        layout.addWidget(self.text_box)

        btn_layout = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet("padding:10px; background-color:#FFA500; color:white; border-radius:8px;")
        back_btn.clicked.connect(lambda: main.go_back(self))
        next_btn = QPushButton("Next → Graphs")
        next_btn.setStyleSheet("padding:10px; background-color:#00CED1; color:white; border-radius:8px;")
        next_btn.clicked.connect(main.show_graphs)
        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(next_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def display_summary(self):
        m = metrics
        cr = m.get("caregiver_ratings", {})
        summary_text = f"""
Basic Metrics:        
- Session Duration: {m.get('session_duration_sec',0):.1f} seconds
- Average Stress Score: {m.get('avg_stress_score',0):.1f}
- Stress Stability: {m.get('stress_stability',0):.1f}
- Overload Events: {m.get('overload_events',0)}

Time in States:
- Calm: {m.get('time_in_states_percent',{}).get('Calm',0):.1f}%
- Mild Stress: {m.get('time_in_states_percent',{}).get('Mild Stress',0):.1f}%
- Overload Risk: {m.get('time_in_states_percent',{}).get('Overload Risk',0):.1f}%

Other Metrics:
- Emotional Balance Index: {m.get('emotional_balance_index',0):.2f}
- Recovery Score: {m.get('recovery_score',0):.2f}
- Average Music Volume: {m.get('avg_music_volume',0):.2f}

Caregiver Feedback Ratings:
- Calmness Level: {cr.get('calmness_level',0)}
- Attention Level: {cr.get('attention_level',0)}
- Willingness to Engage: {cr.get('willingness_to_engage',0)}
- Emotional Response: {cr.get('emotional_response',0)}
- Engagement Duration: {cr.get('engagement_duration',0)}
"""
        self.text_box.setText(summary_text)

class GraphsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QVBoxLayout()
        layout.setContentsMargins(10,10,10,10)
        layout.setSpacing(15)

        self.title = QLabel("📊 Session Graphs")
        self.title.setStyleSheet("font-size:24px; font-weight:bold; color:#00CED1;")
        self.title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        self.canvas_layout = QVBoxLayout()
        layout.addLayout(self.canvas_layout)

        btn_layout = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet("padding:10px; background-color:#FFA500; color:white; border-radius:8px;")
        back_btn.clicked.connect(lambda: main.go_back(self))
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("padding:10px; background-color:#FF6347; color:white; border-radius:8px;")
        close_btn.clicked.connect(sys.exit)
        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def display_graphs(self):
        global session_log
        while self.canvas_layout.count():
            child = self.canvas_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not session_log:
            lbl = QLabel("No session data to plot.")
            self.canvas_layout.addWidget(lbl)
            return

        times = [e["time"]-session_log[0]["time"] for e in session_log]
        stress = [e["stress_score"] for e in session_log]
        volume = [e["volume"] for e in session_log]
        states = [e["state"] for e in session_log]

        fig, axs = plt.subplots(3,1, figsize=(8,6))
        fig.patch.set_facecolor('#2E2E2E')
        fig.tight_layout(pad=4.0)

        axs[0].set_facecolor('#2E2E2E')
        axs[0].plot(times, stress, label='Stress Score')
        axs[0].set_title('Stress Score Over Time', fontsize=14, color='lightcoral')
        axs[0].set_xlabel('Time (sec)', color='white')
        axs[0].set_ylabel('Stress', color='white')
        axs[0].tick_params(colors='white')
        axs[0].grid(True, color='gray', linestyle='--', alpha=0.5)
        axs[0].legend(facecolor='#3E3E3E', edgecolor='white', labelcolor='white')

        axs[1].set_facecolor('#2E2E2E')
        axs[1].plot(times, volume, label='Music Volume')
        axs[1].set_title('Music Volume Over Time', fontsize=14, color='cyan')
        axs[1].set_xlabel('Time (sec)', color='white')
        axs[1].set_ylabel('Volume', color='white')
        axs[1].tick_params(colors='white')
        axs[1].grid(True, color='gray', linestyle='--', alpha=0.5)
        axs[1].legend(facecolor='#3E3E3E', edgecolor='white', labelcolor='white')

        axs[2].set_facecolor('#2E2E2E')
        state_map = {'Calm':0,'Mild Stress':1,'Overload Risk':2}
        state_vals = [state_map.get(s,0) for s in states]
        axs[2].step(times, state_vals, where='post', label='Stress State')
        axs[2].set_yticks([0,1,2])
        axs[2].set_yticklabels(['Calm','Mild','Overload'], color='white')
        axs[2].set_title('Stress States Over Time', fontsize=14, color='lime')
        axs[2].set_xlabel('Time (sec)', color='white')
        axs[2].tick_params(colors='white')
        axs[2].grid(True, color='gray', linestyle='--', alpha=0.5)
        axs[2].legend(facecolor='#3E3E3E', edgecolor='white', labelcolor='white')

        canvas = FigureCanvas(fig)
        self.canvas_layout.addWidget(canvas)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SensoryCoachApp()
    window.show()
    sys.exit(app.exec())



























