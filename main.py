# sensory_coach_with_audio_llm_integration.py
# Full app with continuous bubble waves + robust audio listener + LLM integration
# Requirements: PyQt6, opencv-python, fer, pygame, sounddevice, librosa, numpy,
# matplotlib, huggingface-hub

import sys
import uuid
import time
import threading
import queue
import numpy as np
from fer import FER
from PyQt6 import QtCore, QtGui
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

# --- Hugging Face client ---
import os
import json
from huggingface_hub import InferenceClient

# ---------- CONFIG ----------
# If unsure, run: python -c "import sounddevice as sd; print(sd.query_devices())"
MIC_DEVICE_INDEX = 1   # adjust this if your mic is at a different index
try:
    device_info = sd.query_devices(MIC_DEVICE_INDEX, "input")
    MIC_CHANNELS = device_info.get("max_input_channels", 1) if device_info else 1
except Exception:
    MIC_CHANNELS = 1

session_id = str(uuid.uuid4())
session_start = None
session_log = []
metrics = {}
cap = None
running = False

detector = FER(mtcnn=False)
# initialize pygame mixer carefully - if not available it will print error later
try:
    pygame.mixer.init()
except Exception as e:
    print("pygame.mixer.init() failed:", e)
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
# Audio listener (robust)
# =========================
class AudioListener:
    def __init__(self, device=None, samplerate=16000, block_duration=0.6, channels=1):
        self.samplerate = samplerate
        # slightly shorter block -> faster updates
        self.block_duration = block_duration
        self.blocksize = int(samplerate * block_duration)
        self.device = device
        self.channels = channels
        # larger queue to avoid drops
        self._q = queue.Queue(maxsize=120)
        self._stream = None
        self._thread = None
        self.running = False
        self.lock = Lock()
        self.latest_emotion = "neutral"
        self.latest_score = 0.2
        self.history = collections.deque(maxlen=8)
        self._last_process_time = time.time()

    def _audio_callback(self, indata, frames, time_info, status):
        # Always keep callback fast & robust
        try:
            if status:
                # status is a CallbackFlags with warnings — ignore
                pass
            arr = indata.copy()
            if arr.ndim > 1:
                # mix to mono
                arr = np.mean(arr, axis=1)
            # convert to float32 and scale if it's integer-like
            arr = arr.astype(np.float32)
            maxa = np.max(np.abs(arr)) if arr.size else 0.0
            # if input looks like int16 (values > 1.0), rescale to [-1..1]
            if maxa > 1.0 + 1e-6:
                arr = arr / (maxa + 1e-9)
            try:
                self._q.put_nowait(arr)
            except queue.Full:
                # drop oldest then insert
                try:
                    _ = self._q.get_nowait()
                    self._q.put_nowait(arr)
                except Exception:
                    pass
        except Exception as e:
            # ensure callback never raises
            print("Audio callback error:", e)

    def _processing_loop(self):
        # Keep processing as long as running
        while self.running:
            try:
                block = self._q.get(timeout=0.12)
            except queue.Empty:
                # no audio recently -> keep neutral baseline occasionally
                if time.time() - self._last_process_time > 1.0:
                    with self.lock:
                        self.latest_emotion = "neutral"
                        # keep a small baseline score so UI doesn't show 0.0 permanently
                        self.latest_score = max(0.15, float(self.latest_score * 0.9))
                continue

            try:
                emotion, conf = self._infer_emotion_from_audio(block)
            except Exception as e:
                print("Audio processing error (infer):", e)
                continue

            with self.lock:
                self.history.append((emotion, conf))
                votes = {}
                for emo, c in self.history:
                    votes.setdefault(emo, 0.0)
                    votes[emo] += c
                try:
                    best = max(votes.items(), key=lambda x: x[1])
                    computed = float(min(1.0, votes[best[0]] / (len(self.history) + 1e-6)))
                    # give small floor so it doesn't show 0.0
                    computed = max(0.05, computed)
                    self.latest_emotion = best[0]
                    self.latest_score = computed
                except Exception:
                    self.latest_emotion = "neutral"
                    self.latest_score = 0.15
            self._last_process_time = time.time()
            # tiny sleep to be cooperative (processing driven by queue)
            time.sleep(0.005)

    def _infer_emotion_from_audio(self, audio_block):
        """
        audio_block: 1D float32 array in approx [-1..1]
        returns: (label, confidence)
        """
        audio = audio_block.astype(np.float32).flatten()
        if audio.size == 0:
            return ("neutral", 0.15)

        # ensure not all zeros
        max_abs = np.max(np.abs(audio)) + 1e-9
        if max_abs > 0:
            audio = audio / max_abs

        # compute rms robustly
        try:
            rms = float(np.sqrt(np.mean(audio**2)))
        except Exception:
            rms = 0.0

        # fallback feature values
        centroid = 0.0
        zcr = 0.0
        mfcc0 = 0.0

        # use librosa if it works for this block length
        try:
            # librosa requires at least a few frames for some features; guard with try
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.samplerate))) if audio.size >= 64 else 0.0
        except Exception:
            centroid = 0.0

        try:
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio))) if audio.size >= 16 else float(np.mean(np.abs(np.diff(np.sign(audio))))) if audio.size >= 2 else 0.0
        except Exception:
            zcr = float(np.mean(np.abs(np.diff(np.sign(audio))))) if audio.size >= 2 else 0.0

        try:
            if audio.size >= 256:
                mfcc = librosa.feature.mfcc(y=audio, sr=self.samplerate, n_mfcc=13)
                mfcc0 = float(np.mean(mfcc[0]))
            else:
                mfcc0 = float(np.mean(audio) * 100.0)
        except Exception:
            mfcc0 = float(np.mean(audio) * 100.0)

        # Normalize features into reasonable ranges
        act_norm = min(1.0, rms * 25.0)   # amplify sensitivity
        bright_norm = min(1.0, centroid / (self.samplerate / 4 + 1e-9)) if centroid > 0 else 0.0
        zcr_norm = min(1.0, zcr * 5.0)

        angry_score = max(0.0, act_norm * 0.6 + bright_norm * 0.3 + zcr_norm * 0.1)
        sad_score = max(0.0, (1 - act_norm) * 0.6 + (1 - bright_norm) * 0.3)
        neutral_score = max(0.0, 1 - abs(act_norm - 0.4) - abs(bright_norm - 0.35))
        happy_score = max(0.0, 0.5 * act_norm + 0.5 * (1 - abs(mfcc0) / 50.0))
        surprise_score = max(0.0, 0.8 * act_norm * bright_norm)

        score_map = {
            "angry": angry_score,
            "sad": sad_score,
            "neutral": neutral_score,
            "happy": happy_score,
            "surprise": surprise_score
        }
        label, raw_score = max(score_map.items(), key=lambda x: x[1])
        conf = float(min(1.0, raw_score))
        # If confidence tiny, give neutral a small boost to avoid zeros
        if conf < 0.10:
            label = "neutral"
            conf = max(conf, 0.12)
        return (label, conf)

    def start(self):
        if self.running:
            return
        self.running = True

        # attempt to open stream robustly with fallbacks
        attempts = []
        tried = set()

        def try_open(device, channels):
            key = (device if device is not None else "default", channels)
            if key in tried:
                return False
            tried.add(key)
            try:
                self._stream = sd.InputStream(
                    samplerate=self.samplerate,
                    device=device,
                    channels=channels,
                    blocksize=self.blocksize,
                    callback=self._audio_callback
                )
                self._stream.start()
                self.device = device
                self.channels = channels
                print(f"Audio stream started on device {device} with {channels} channel(s)")
                return True
            except Exception as e:
                attempts.append((device, channels, str(e)))
                return False

        # try configured device
        if try_open(self.device, self.channels):
            pass
        else:
            # try default device with same channels
            try:
                default_dev = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            except Exception:
                default_dev = None
            if try_open(default_dev, self.channels):
                pass
            else:
                # try mono on configured device
                if try_open(self.device, 1):
                    pass
                else:
                    # try mono on default
                    if try_open(default_dev, 1):
                        pass
                    else:
                        # scan devices to find an input device supporting at least 1 channel
                        try:
                            devs = sd.query_devices()
                            found = False
                            for idx, d in enumerate(devs):
                                try:
                                    mic_ch = d.get('max_input_channels', 0)
                                except Exception:
                                    mic_ch = 0
                                if mic_ch >= 1:
                                    if try_open(idx, 1):
                                        found = True
                                        break
                            if not found:
                                raise RuntimeError("No suitable input device found")
                        except Exception as e:
                            print("Audio input stream error (final):", e)
                            print("Audio attempts:", attempts)
                            self.running = False
                            return

        # start processing thread
        self._thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
        except Exception:
            pass
        try:
            with self._q.mutex:
                self._q.queue.clear()
        except Exception:
            pass

# global audio listener
audio_listener = AudioListener(device=MIC_DEVICE_INDEX, samplerate=16000, block_duration=0.6, channels=MIC_CHANNELS)

# =========================
# LLM integration utilities (unchanged)
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN")
_client = None
if HF_TOKEN:
    try:
        _client = InferenceClient(api_key=HF_TOKEN)
    except Exception as e:
        print("Warning: could not create HF client:", e)
else:
    print("Warning: HF_TOKEN not set in environment. LLM features disabled until HF_TOKEN is provided.")

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def _format_metrics_for_prompt(metrics: dict, max_chars_for_json: int = 4000):
    try:
        pretty_json = json.dumps(metrics, indent=2, ensure_ascii=False)
    except Exception:
        def sanitize(o):
            try:
                return float(o)
            except Exception:
                return str(o)
        pretty_json = json.dumps(metrics, indent=2, default=sanitize, ensure_ascii=False)

    if len(pretty_json) > max_chars_for_json:
        pretty_json = pretty_json[:max_chars_for_json] + "\n  ... (truncated) ...\n"

    summary_lines = []
    summary_lines.append(f"Session ID: {metrics.get('session_id','N/A')}")
    duration = metrics.get("session_duration_sec", 0.0)
    summary_lines.append(f"Duration (sec): {duration:.1f}")
    summary_lines.append(f"Avg Stress Score: {metrics.get('avg_stress_score',0):.2f}")
    summary_lines.append(f"Stress Stability (std): {metrics.get('stress_stability',0):.2f}")
    summary_lines.append(f"Overload Events: {metrics.get('overload_events',0)}")
    tstates = metrics.get("time_in_states_percent", {})
    summary_lines.append("Time in states (%, Calm/Mild/Overload): "
                         f"{tstates.get('Calm',0):.1f}/{tstates.get('Mild Stress',0):.1f}/{tstates.get('Overload Risk',0):.1f}")
    ebal = metrics.get("emotional_balance_index", None)
    if ebal is not None:
        summary_lines.append(f"Emotional balance index: {ebal:.2f}")
    summary_lines.append(f"Avg music volume: {metrics.get('avg_music_volume',0):.2f}")

    cr = metrics.get("caregiver_ratings", {})
    if cr:
        cr_summary = ", ".join([f"{k}:{v}" for k,v in cr.items()])
        summary_lines.append(f"Caregiver ratings: {cr_summary}")

    human_summary = "\n".join(summary_lines)
    return pretty_json, human_summary

def _build_messages(metrics_json: str, human_summary: str, instruction: str = None):
    """
    Returns a list of messages for HF chat completion (single user message containing the task + data).
    """
    screenshot_file_url = "file:///mnt/data/08186969-b385-43ce-9aed-a8fc1683de81.png"
    combined = (
        "System Instructions:\n"
        "- You are a clinician-facing therapy assistant specializing in Autism Spectrum Conditions.\n"
        "- Your tone must be calm, concise, and professional.\n"
        "- Your answer must be clean, readable, and formatted with proper headings and bullet points.\n"
        "- DO NOT output JSON, code blocks, or any technical formatting.\n"
        f"- DO NOT mention or describe any image attachment. (Attachment is for tooling only: {screenshot_file_url})\n\n"
        "Task:\n"
        "Based on the human summary and raw metrics below, produce **TWO SECTIONS ONLY**:\n\n"
        "1) **Identified Weakness (1–2 sentences)**\n"
        "   - A short, plain-language description of the child's main challenge inferred from the metrics.\n\n"
        "2) **Recommended In-Person Therapy Activity (Well-Formatted)**\n"
        "   Provide exactly ONE activity that directly targets the weakness above.\n"
        "   Format MUST include the following clear headings:\n"
        "   • **Title**\n"
        "   • **Age Range**\n"
        "   • **Materials**\n"
        "   • **Environment Adjustments** (bullet points)\n"
        "   • **Steps** (3–6 short numbered steps)\n"
        "   • **Caregiver Script** (1–2 short sentences)\n"
        "   • **Duration** (minutes)\n"
        "   • **Rationale** (1–2 sentences)\n"
        "   • **Safety Notes / Adaptations**\n"
        "   • **Priority** (High / Medium / Low)\n\n"
        "RULES:\n"
        "- Provide **only one weakness** and **only one activity**.\n"
        "- No unnecessary text, no extra sections.\n"
        "- If the metrics are insufficient to infer a weakness, reply with:\n"
        "  \"I cannot recommend an in-person therapy activity because ...\"\n"
        "  followed by your short reason.\n"
        "- The final answer MUST be easy to read with clean spacing.\n\n"
        "Refusal Criteria (MUST follow strictly):\n"
        "- If session duration < 40 seconds\n"
        "- OR avg stress = 0 AND stress stability < 1\n"
        "- OR no emotional distribution available\n"
        "- OR all time-in-states percentages are 0\n"
        "- OR fewer than 5 stress log entries\n"
        "- OR caregiver ratings missing AND metrics show no variation\n"
        "→ Then output ONLY this single sentence:\n"
        "  \"I cannot recommend an in-person therapy activity because ...\"\n"
        "  followed by the reason.\n\n"
    )

    if instruction:
        combined += f"Additional Instruction: {instruction.strip()}\n\n"

    # Append the human summary + raw metrics JSON at the end
    combined += (
        "Human Summary:\n"
        f"{human_summary}\n\n"
        "Raw Metrics JSON (for analysis only):\n"
        f"{metrics_json}\n"
    )

    # Return as a single user message (as expected by send_metrics_to_llm)
    return [{"role": "user", "content": combined}]

def send_metrics_to_llm(metrics: dict, instruction: str = None, model: str = DEFAULT_MODEL,
                        temperature: float = 0.1):
    """
    Synchronous LLM call helper. Returns assistant string or an error message.
    Caller should call this in a background thread to avoid blocking the GUI.
    """
    if _client is None:
        return "[LLM disabled] HF_TOKEN not available or client could not be created."

    metrics_json, human_summary = _format_metrics_for_prompt(metrics)
    messages = _build_messages(metrics_json, human_summary, instruction)

    try:
        completion = _client.chat.completions.create(
            model=model,
            messages=messages,
        )
        assistant_message = None
        try:
            assistant_message = completion.choices[0].message.get("content")
        except Exception:
            assistant_message = str(completion)
        return assistant_message
    except Exception as e:
        return f"[LLM request failed] {e}"

# =========================
# Bubble Pop Game Widget (continuous spawn improvements)
# =========================
class BubbleGameWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.bubbles = []
        self._lock = threading.Lock()
        self.game_timer = QtCore.QTimer()
        self.game_timer.setInterval(30)
        self.game_timer.timeout.connect(self.update_game)
        self.game_timer.start()

        # more frequent spawn for continuous waves
        self.spawn_timer = QtCore.QTimer()
        self.spawn_timer.setInterval(450)  # ms
        self.spawn_timer.timeout.connect(self.spawn_bubble)
        self.spawn_timer.start()

        # overlay data (updated by MonitorPage)
        self.stress_score = 0
        self.state = "Calm"
        self.color = (0, 255, 255)
        self.new_volume = 0.3
        self.disp_video_emotion = "-"
        self.disp_video_score = 0.0
        self.disp_audio_emotion = "-"
        self.disp_audio_score = 0.0
        self.message = messages.get(self.state, "")
        try:
            self.pop_sound = pygame.mixer.Sound("pop.wav")
        except Exception:
            self.pop_sound = None

        self._last_spawn_time = time.time()
        self._max_bubbles = 400  # safety cap

    def spawn_bubble(self):
        # spawn bubbles based on current stress (lower stress => more bubbles)
        w = max(200, self.width())
        h = max(200, self.height())
        with self._lock:
            stress = float(getattr(self, "stress_score", 30))
            # Guarantee at least 1 bubble, up to 6 depending on stress
            if stress < 20:
                count = np.random.randint(3, 7)
            elif stress < 40:
                count = np.random.randint(2, 5)
            elif stress < 60:
                count = np.random.randint(1, 4)
            elif stress < 80:
                count = np.random.randint(1, 3)
            else:
                count = 1
            # limit total bubbles to avoid runaway
            available = max(0, self._max_bubbles - len(self.bubbles))
            count = min(count, available) if available > 0 else 0
            for _ in range(count):
                x = float(np.random.randint(30, max(31, w - 30)))
                radius = float(np.random.randint(16, 46))
                speed = float(np.random.uniform(0.6, 3.2))
                color = (np.random.randint(80,255), np.random.randint(80,255), np.random.randint(80,255))
                # place slightly off bottom so they float in
                self.bubbles.append({
                    "x": x,
                    "y": float(h + radius + np.random.uniform(0, 40)),
                    "r": radius,
                    "speed": speed,
                    "color": color,
                    "created": time.time()
                })
            self._last_spawn_time = time.time()

    def update_game(self):
        now = time.time()
        with self._lock:
            # move bubbles upward and apply slight horizontal jitter
            for b in self.bubbles:
                jitter = np.sin((now + b["x"]) * 0.6) * 0.6
                b["x"] += jitter * 0.3
                b["y"] -= b["speed"]
            # remove off-screen (and also very old ones)
            self.bubbles = [b for b in self.bubbles if (b["y"] + b["r"] > 0 and (now - b.get("created", now) < 60))]
            # safety: if spawn timer hasn't fired for some reason, spawn a tiny wave
            interval_seconds = max(0.2, self.spawn_timer.interval() / 1000.0)
            if (now - self._last_spawn_time) > interval_seconds * 1.4:
                # spawn at least one bubble to keep motion alive
                if len(self.bubbles) < self._max_bubbles:
                    w = max(200, self.width())
                    h = max(200, self.height())
                    x = float(np.random.randint(30, max(31, w - 30)))
                    radius = float(np.random.randint(14, 40))
                    speed = float(np.random.uniform(0.8, 2.4))
                    color = (np.random.randint(80,255), np.random.randint(80,255), np.random.randint(80,255))
                    self.bubbles.append({"x": x, "y": float(h + radius), "r": radius, "speed": speed, "color": color, "created": time.time()})
                    self._last_spawn_time = now
            # keep minimum wave activity: if there are no bubbles, create a burst
            if len(self.bubbles) == 0:
                # quick burst to restart motion
                for _ in range(3):
                    w = max(200, self.width())
                    h = max(200, self.height())
                    x = float(np.random.randint(30, max(31, w - 30)))
                    radius = float(np.random.randint(16, 36))
                    speed = float(np.random.uniform(0.8, 2.6))
                    color = (np.random.randint(80,255), np.random.randint(80,255), np.random.randint(80,255))
                    self.bubbles.append({"x": x, "y": float(h + radius), "r": radius, "speed": speed, "color": color, "created": time.time()})
        # request repaint
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Background
        qp.fillRect(self.rect(), QtGui.QColor(30,30,30))

        # Draw bubbles
        with self._lock:
            for b in list(self.bubbles):
                color = QtGui.QColor(int(b["color"][0]), int(b["color"][1]), int(b["color"][2]))
                qp.setBrush(color)
                qp.setPen(QtCore.Qt.PenStyle.NoPen)
                qp.drawEllipse(int(b["x"] - b["r"]), int(b["y"] - b["r"]), int(2*b["r"]), int(2*b["r"]))

        # Draw analytics overlay (top)
        w = self.width()
        h = self.height()
        overlay_h = 100
        qp.setBrush(QtGui.QColor(20,20,20,220))
        qp.setPen(QtCore.Qt.PenStyle.NoPen)
        qp.drawRect(0, 0, w, overlay_h)

        # Title
        qp.setPen(QtGui.QColor(0,255,255))
        font = QtGui.QFont("Segoe UI", 14, QtGui.QFont.Weight.Bold)
        qp.setFont(font)
        qp.drawText(20, 28, "SensoryCoach - Stress Monitor")

        # State pill
        pill_color = QtGui.QColor(int(self.color[0]), int(self.color[1]), int(self.color[2]))
        qp.setBrush(pill_color)
        qp.setPen(QtCore.Qt.PenStyle.NoPen)
        qp.drawRoundedRect(20, 60, 180, 28, 6, 6)
        qp.setPen(QtGui.QColor(0,0,0))
        font2 = QtGui.QFont("Segoe UI", 10, QtGui.QFont.Weight.Bold)
        qp.setFont(font2)
        qp.drawText(30, 80, self.state)

        # Stress bar
        bar_x, bar_y, bar_w, bar_h = 220, 70, 300, 20
        qp.setPen(QtGui.QColor(100,100,100))
        qp.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        qp.drawRect(bar_x, bar_y, bar_w, bar_h)
        qp.setBrush(pill_color)
        filled_w = int(bar_w * (max(0, min(100, self.stress_score))/100.0))
        qp.drawRect(bar_x, bar_y, filled_w, bar_h)
        qp.setPen(QtGui.QColor(255,255,255))
        qp.setFont(QtGui.QFont("Segoe UI", 9))
        qp.drawText(bar_x + bar_w + 20, bar_y + 15, f"Stress: {int(self.stress_score)}")

        # Volume bar
        vol_x, vol_y, vol_w, vol_h = 560, 70, 150, 20
        qp.setPen(QtGui.QColor(100,100,100))
        qp.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        qp.drawRect(vol_x, vol_y, vol_w, vol_h)
        qp.setBrush(QtGui.QColor(0,150,255))
        qp.drawRect(vol_x, vol_y, int(vol_w * max(0, min(1, self.new_volume))), vol_h)
        qp.setPen(QtGui.QColor(255,255,255))
        qp.drawText(vol_x + vol_w + 20, vol_y + 15, f"Volume: {self.new_volume:.2f}")

        # Bottom overlay for emotions & message
        qp.setBrush(QtGui.QColor(20,20,20,220))
        qp.setPen(QtCore.Qt.PenStyle.NoPen)
        qp.drawRect(0, h-80, w, 80)
        qp.setPen(QtGui.QColor(255,255,255))
        qp.setFont(QtGui.QFont("Segoe UI", 10))
        disp_line = f"Video: {self.disp_video_emotion} ({self.disp_video_score:.2f})  |  Audio: {self.disp_audio_emotion} ({self.disp_audio_score:.2f})"
        qp.drawText(30, h-40, disp_line)
        qp.setFont(QtGui.QFont("Segoe UI", 12, QtGui.QFont.Weight.Bold))
        qp.drawText(30, h-18, self.message)

        qp.end()

    def mousePressEvent(self, event):
        mx = event.position().x()
        my = event.position().y()
        popped = None
        with self._lock:
            for b in list(self.bubbles):
                if (mx - b["x"])**2 + (my - b["y"])**2 <= b["r"]**2:
                    popped = b
                    try:
                        self.bubbles.remove(b)
                    except Exception:
                        pass
                    break
        if popped and self.pop_sound:
            try:
                self.pop_sound.play()
            except Exception:
                pass
        # Optional: reward / change stress slightly on popping
        if popped:
            try:
                self.stress_score = max(0, self.stress_score - 0.5)
            except Exception:
                pass
        super().mousePressEvent(event)

    def update_overlay(self, stress_score, state, color, new_volume,
                       disp_video_emotion, disp_video_score, disp_audio_emotion, disp_audio_score, message):
        # called by MonitorPage to update overlay values
        self.stress_score = float(stress_score)
        self.state = state
        self.color = color
        self.new_volume = float(new_volume)
        self.disp_video_emotion = disp_video_emotion
        self.disp_video_score = disp_video_score
        self.disp_audio_emotion = disp_audio_emotion
        self.disp_audio_score = disp_audio_score
        self.message = message
        # optionally tune spawn rate by stress: lower stress -> faster spawn interval
        try:
            if self.stress_score < 25:
                self.spawn_timer.setInterval(400)
            elif self.stress_score < 50:
                self.spawn_timer.setInterval(700)
            else:
                self.spawn_timer.setInterval(1200)
        except Exception:
            pass
        self.update()

# =========================
# GUI (mostly as original) with audio integration improvements + Recommendations page
# =========================

def combine_scores(video_emotion, video_score, audio_emotion, audio_score):
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
        # instantiate RecommendationsPage instead of LLMResponsePage
        self.recommendations_page = RecommendationsPage(self)

        for page in [self.start_page, self.monitor_page, self.feedback_page, self.summary_page, self.graphs_page, self.recommendations_page]:
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

    def show_llm(self):
        # keep method name for compatibility with existing callbacks
        self.stack.setCurrentWidget(self.recommendations_page)
        self.recommendations_page.refresh()

    def go_back(self, current_page):
        if current_page == self.monitor_page:
            self.show_start()
        elif current_page == self.feedback_page:
            self.show_monitor()
        elif current_page == self.summary_page:
            self.show_feedback()
        elif current_page == self.graphs_page:
            self.show_summary()
        elif current_page == self.recommendations_page:
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

        # show feedback page (user can then go to summary and then View Recommendations)
        self.show_feedback()

# ---------- UI Pages (copied from your original, unchanged behavior) ----------
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

        # replaced camera label with bubble game widget
        self.bubble_widget = BubbleGameWidget()
        self.bubble_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.bubble_widget, stretch=1)

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

        # Read a frame from camera for face/emotion analytics (we do NOT display the frame)
        ret, frame = cap.read()
        if not ret:
            # if camera fails, still update overlay from audio only
            frame = None

        h_label = self.bubble_widget.height()
        w_label = self.bubble_widget.width()
        # protect against tiny widget sizes
        if h_label < 50 or w_label < 50:
            h_label, w_label = 480, 640

        self.frame_counter += 1

        video_emotion, video_score = (None, 0.0)
        audio_emotion, audio_score = (None, 0.0)

        # Video: sample every N frames for performance (if frame available)
        if self.frame_counter % 10 == 0 and frame is not None:
            # optionally run detection on a smaller frame
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

        # Prepare display values (use last-known so 0.00 doesn't stick)
        disp_video_emotion = (video_emotion or self.last_video_emotion or "-")
        disp_video_score = (video_score or self.last_video_score or 0.0)
        disp_audio_emotion = (audio_emotion or self.last_audio_emotion or "-")
        disp_audio_score = (audio_score or self.last_audio_score or 0.0)

        # Update bubble widget overlay (replaces setting camera pixmap)
        self.bubble_widget.update_overlay(
            stress_score=self.stress_score,
            state=self.state,
            color=self.color,
            new_volume=self.new_volume,
            disp_video_emotion=disp_video_emotion,
            disp_video_score=disp_video_score,
            disp_audio_emotion=disp_audio_emotion,
            disp_audio_score=disp_audio_score,
            message=messages.get(self.state, "")
        )

class CaregiverInsightsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QVBoxLayout()
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(12)

        title = QLabel("\ud83e\udd0d Caregiver Feedback")
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

        self.title = QLabel("\u2728 Session Metrics \u2728")
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
        ask_llm_btn = QPushButton("View Recommendations")
        ask_llm_btn.setStyleSheet("padding:10px; background-color:#6A5ACD; color:white; border-radius:8px;")
        ask_llm_btn.clicked.connect(main.show_llm)
        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(next_btn)
        btn_layout.addWidget(ask_llm_btn)
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

        self.title = QLabel("\ud83d\udcca Session Graphs")
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

class RecommendationsPage(QWidget):
    # signal to safely send text back to main thread
    result_signal = QtCore.pyqtSignal(str)

    def __init__(self, main):
        super().__init__()
        self.main = main
        layout = QVBoxLayout()
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(10)

        # Title no longer mentions LLM
        title = QLabel("✨ Recommendations")
        title.setStyleSheet("font-size:22px; font-weight:bold; color:#00CED1;")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Short helper text
        self.info = QLabel("Click 'Get Recommendations' to generate personalized session recommendations.")
        self.info.setStyleSheet("font-size:14px; color:#EEEEEE;")
        self.info.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setStyleSheet(
            "font-size:14px; color:#FFFFFF; background-color:#222222; padding:12px; border-radius:8px;"
        )
        layout.addWidget(self.output, stretch=1)

        btn_layout = QHBoxLayout()
        # new button label
        self.query_btn = QPushButton("Get Recommendations")
        self.query_btn.setStyleSheet("padding:10px; background-color:#6A5ACD; color:white; border-radius:8px;")
        self.query_btn.clicked.connect(self.on_query)
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet("padding:10px; background-color:#FFA500; color:white; border-radius:8px;")
        back_btn.clicked.connect(lambda: main.go_back(self))
        btn_layout.addWidget(back_btn)
        btn_layout.addWidget(self.query_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.result_signal.connect(self._on_result)

    def refresh(self):
        self.output.clear()

    def on_query(self):
        # Disable the button and clear previous content. Do NOT show extra status lines.
        self.query_btn.setDisabled(True)
        self.output.clear()
        # start background worker thread
        t = threading.Thread(target=self._worker_call_llm, daemon=True)
        t.start()

    def _worker_call_llm(self):
        try:
            res = send_metrics_to_llm(metrics)
        except Exception as e:
            res = f"[Model call error] {e}"
        # emit back to UI thread
        self.result_signal.emit(str(res))

    def _beautify_text(self, raw: str) -> str:
        """
        Try to present the model response cleanly:
        - If valid JSON, pretty-print it into sections.
        - Else try to split by numbered sections or newlines, otherwise wrap into paragraphs.
        """
        # 1) Try JSON
        try:
            parsed = json.loads(raw)
            # If it's a dict with expected keys, format nicely
            out_lines = []
            if isinstance(parsed, dict):
                # common keys heuristically handled
                if "interpretation" in parsed:
                    out_lines.append("Interpretation:\n" + parsed.get("interpretation","").strip())
                if "concerns" in parsed:
                    out_lines.append("\nConcerns:")
                    for c in parsed.get("concerns",[]):
                        out_lines.append(f"• {c}")
                if "recommendations" in parsed:
                    out_lines.append("\nRecommendations:")
                    for i, r in enumerate(parsed.get("recommendations",[]),1):
                        out_lines.append(f"{i}. {r}")
                if "micro_intervention" in parsed:
                    out_lines.append("\nMicro-intervention:\n" + parsed.get("micro_intervention","").strip())
                # fallback: dump remaining keys
                other_keys = set(parsed.keys()) - {"interpretation","concerns","recommendations","micro_intervention"}
                for k in other_keys:
                    out_lines.append(f"\n{k}:\n{json.dumps(parsed[k], indent=2)}")
                return "\n".join(out_lines).strip()
            else:
                # not a dict: pretty-print JSON value
                return json.dumps(parsed, indent=2)
        except Exception:
            pass

        # 2) Not JSON — try to split into sections by common headings (heuristic)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            return raw.strip()

        # If it contains numbered list style, keep it but add headings
        joined = "\n".join(lines)
        # Try to separate Interpretation / Concerns / Recommendations by keywords
        for key in ("Interpretation:", "Interpretation", "Concerns:", "Concerns", "Recommendations:", "Recommendations",
                    "Micro-intervention", "Micro intervention", "Micro-intervention:"):
            if key.lower() in joined.lower():
                # insert blank line before headings for clarity
                return joined

        # Otherwise, produce a simple structured view:
        result = []
        # first 1-2 lines as short interpretation
        result.append("Interpretation:\n" + (" ".join(lines[:2]) if len(lines)>=2 else lines[0]))
        # rest as recommendations / details
        if len(lines) > 2:
            result.append("\nDetails / Suggestions:")
            for ln in lines[2:]:
                result.append(f"• {ln}")
        return "\n".join(result)

    def _on_result(self, text: str):
       # Beautify formatting
        beaut = self._beautify_text(text)

        # Manual HTML escape (Qt6 safe)
        def esc(s):
            return (
                s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        html = f"""
        <div style='color:#FFFFFF; font-family:Segoe UI, Roboto, Arial, sans-serif;
                    line-height:1.5; padding:12px;'>

            <h2 style='color:#00CED1; text-align:center; margin-bottom:18px;'>
                ✨ Recommendations
            </h2>

            <div style='background:#1e1e1e; padding:16px; border-radius:10px;
                        font-size:14px; color:#f0f0f0;'>
                <pre style='white-space:pre-wrap; font-family:inherit; font-size:14px;'>
    {esc(beaut)}
                </pre>
            </div>

        </div>
        """

        try:
            self.output.setHtml(html)
        except Exception:
            # fallback
            self.output.setPlainText(beaut)

        self.query_btn.setDisabled(False)

# ---------- main ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SensoryCoachApp()
    window.show()
    sys.exit(app.exec())
