import sys
import uuid
import time
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

def get_stress_score(emotion, score):
    if emotion in ["happy", "neutral"]:
        return -15 * score
    elif emotion == "surprise":
        return 10 * score
    elif emotion in ["fear", "angry", "sad", "disgust"]:
        return 20 * score
    return 0

def start_music(volume=0.3):
    pygame.mixer.music.load(MUSIC_FILE)
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play(-1)

def set_music_volume(volume):
    pygame.mixer.music.set_volume(volume)

def stop_music():
    pygame.mixer.music.stop()

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
        global cap, running, session_start, session_log
        session_log = []
        session_start = time.time()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        running = True
        start_music(0.3)
        self.monitor_page.start_timer()

    def finish_session(self):
        global running, metrics, session_log, cap, session_start
        running = False
        if cap:
            cap.release()
        stop_music()

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

    def start_timer(self):
        self.timer.start(30)

    def update_frame(self):
        global running, session_log, cap
        if not running:
            self.timer.stop()
            return
        ret, frame = cap.read()
        if not ret:
            return

        h_label = self.label.height()
        w_label = self.label.width()
        frame = cv2.resize(frame, (w_label, int(h_label*0.9)))

        self.frame_counter += 1

        if self.frame_counter % 10 == 0:
            small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            emotions = detector.detect_emotions(small_frame)
            if emotions:
                self.dominant_emotion, self.emo_score = detector.top_emotion(small_frame)
                self.stress_score += get_stress_score(self.dominant_emotion, self.emo_score)

        self.stress_score = max(0, min(100, self.stress_score))

        if self.stress_score < thresholds["calm"]:
            self.state = "Calm"
            self.color = (0, 255, 255)
            self.new_volume = 0.3
        elif self.stress_score < thresholds["mild"]:
            self.state = "Mild Stress"
            self.color = (0, 255, 0)
            self.new_volume = min(0.7, 0.3+self.emo_score)
        else:
            self.state = "Overload Risk"
            self.color = (0, 0, 255)
            self.new_volume = min(1.0, 0.5+self.emo_score)

        set_music_volume(self.new_volume)

        session_log.append({
            "session_id": session_id,
            "time": time.time(),
            "emotion": self.dominant_emotion,
            "score": float(self.emo_score),
            "stress_score": float(self.stress_score),
            "state": self.state,
            "volume": float(self.new_volume)
        })

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

        cv2.rectangle(frame, (0, h-50), (w,h), (20,20,20), -1)
        cv2.putText(frame, messages[self.state], (30, h-20), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)

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

        times = [e["time"]-session_log[0]["time"] for e in session_log]
        stress = [e["stress_score"] for e in session_log]
        volume = [e["volume"] for e in session_log]
        states = [e["state"] for e in session_log]

        fig, axs = plt.subplots(3,1, figsize=(8,6))
        fig.patch.set_facecolor('#2E2E2E')
        fig.tight_layout(pad=4.0)

        axs[0].set_facecolor('#2E2E2E')
        axs[0].plot(times, stress, color='red', label='Stress Score')
        axs[0].set_title('Stress Score Over Time', fontsize=14, color='lightcoral')
        axs[0].set_xlabel('Time (sec)', color='white')
        axs[0].set_ylabel('Stress', color='white')
        axs[0].tick_params(colors='white')
        axs[0].grid(True, color='gray', linestyle='--', alpha=0.5)
        axs[0].legend(facecolor='#3E3E3E', edgecolor='white', labelcolor='white')

        axs[1].set_facecolor('#2E2E2E')
        axs[1].plot(times, volume, color='cyan', label='Music Volume')
        axs[1].set_title('Music Volume Over Time', fontsize=14, color='cyan')
        axs[1].set_xlabel('Time (sec)', color='white')
        axs[1].set_ylabel('Volume', color='white')
        axs[1].tick_params(colors='white')
        axs[1].grid(True, color='gray', linestyle='--', alpha=0.5)
        axs[1].legend(facecolor='#3E3E3E', edgecolor='white', labelcolor='white')

        axs[2].set_facecolor('#2E2E2E')
        state_map = {'Calm':0,'Mild Stress':1,'Overload Risk':2}
        state_vals = [state_map[s] for s in states]
        axs[2].step(times, state_vals, where='post', color='lime', label='Stress State')
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
























