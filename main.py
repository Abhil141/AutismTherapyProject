import cv2
from fer import FER
import numpy as np
import time
import json
import pygame
import uuid
import matplotlib.pyplot as plt

# ---------------------------
# Session Setup
# ---------------------------
session_id = str(uuid.uuid4())  # unique session ID
session_start = time.time()

# Initialize FER detector (lighter without mtcnn for speed)
detector = FER(mtcnn=False)

# Initialize pygame mixer for soothing music
pygame.mixer.init()
pygame.mixer.music.load("soothing_music.mp3")
pygame.mixer.music.play(-1)  # loop forever
pygame.mixer.music.set_volume(0.3)

# Open webcam
cap = cv2.VideoCapture(0)

# Adaptive stress thresholds
thresholds = {"calm": 30, "mild": 60}
stress_score = 0

# Log storage
session_log = []

# Encouraging messages
messages = {
    "Calm": "Great job staying relaxed!",
    "Mild Stress": "You're doing fine. Take a deep breath.",
    "Overload Risk": "Pause for a moment, you've got this."
}

# ---------------------------
# Helper Function
# ---------------------------
def get_stress_score(emotion, score):
    if emotion in ["happy", "neutral"]:
        return -15 * score
    elif emotion == "surprise":
        return 10 * score
    elif emotion in ["fear", "angry", "sad", "disgust"]:
        return 20 * score
    else:
        return 0

# ---------------------------
# Splash Screen
# ---------------------------
splash = np.zeros((500, 800, 3), dtype=np.uint8)
cv2.putText(splash, "=== Welcome to SensoryCoach ===", (60, 200),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
cv2.putText(splash, "Sit comfortably, breathe deeply.", (140, 270),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
cv2.putText(splash, "Look at the camera. Game starts soon!", (100, 320),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
cv2.putText(splash, "Press 'q' anytime to exit.", (180, 370),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
cv2.imshow("SensoryCoach", splash)
cv2.waitKey(3000)  # Show splash for 3 seconds

# ---------------------------
# Main Loop
# ---------------------------
frame_counter = 0
dominant_emotion, emo_score = None, 0
state = "Calm"
color = (0, 255, 255)
new_volume = 0.3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # Run emotion detection every 10 frames
    if frame_counter % 10 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        emotions = detector.detect_emotions(small_frame)
        if emotions:
            dominant_emotion, emo_score = detector.top_emotion(small_frame)
            stress_score += get_stress_score(dominant_emotion, emo_score)

    # Clamp stress
    stress_score = max(0, min(100, stress_score))

    # Determine state
    if stress_score < thresholds["calm"]:
        state = "Calm"
        color = (0, 255, 255)
        new_volume = 0.3
    elif stress_score < thresholds["mild"]:
        state = "Mild Stress"
        color = (0, 255, 0)
        new_volume = min(0.7, 0.3 + emo_score)
    else:
        state = "Overload Risk"
        color = (0, 0, 255)
        new_volume = min(1.0, 0.5 + emo_score)

    # Update music volume
    pygame.mixer.music.set_volume(new_volume)

    # Log
    session_log.append({
        "session_id": session_id,
        "time": time.time(),
        "emotion": dominant_emotion,
        "score": float(emo_score),
        "stress_score": float(stress_score),
        "state": state,
        "volume": float(new_volume)
    })

    # --- HUD Overlay ---
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.putText(frame, "SensoryCoach - Stress Monitor", (20, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    cv2.rectangle(frame, (20, 60), (200, 100), color, -1)
    cv2.putText(frame, state, (30, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)

    # Stress bar
    bar_x, bar_y, bar_w, bar_h = 220, 70, 300, 20
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 2)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * (stress_score / 100)), bar_y + bar_h), color, -1)
    cv2.putText(frame, f"Stress: {int(stress_score)}", (bar_x + bar_w + 20, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Volume bar
    vol_x, vol_y, vol_w, vol_h = 560, 70, 150, 20
    cv2.rectangle(frame, (vol_x, vol_y), (vol_x + vol_w, vol_y + vol_h), (100, 100, 100), 2)
    cv2.rectangle(frame, (vol_x, vol_y),
                  (vol_x + int(vol_w * new_volume), vol_y + vol_h), (0, 150, 255), -1)
    cv2.putText(frame, f"Volume: {new_volume:.2f}", (vol_x + vol_w + 20, vol_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Encouraging message
    cv2.rectangle(frame, (0, h - 60), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, messages[state], (30, h - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    # Show
    cv2.imshow("SensoryCoach", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()

# ---------------------------
# Save raw log
# ---------------------------
with open(f"session_log_{session_id}.json", "w") as f:
    json.dump(session_log, f, indent=2)

# ---------------------------
# Therapy & Clinical Metrics
# ---------------------------
session_end = time.time()
duration = session_end - session_start
stress_values = [e["stress_score"] for e in session_log]
states = [e["state"] for e in session_log]
emotions = [e["emotion"] for e in session_log if e["emotion"]]
volumes = [e["volume"] for e in session_log]

# Stress metrics
avg_stress = float(np.mean(stress_values)) if stress_values else 0
std_stress = float(np.std(stress_values)) if stress_values else 0
overload_events = states.count("Overload Risk")
time_in_states = {
    "Calm": states.count("Calm") / max(1, len(states)) * 100,
    "Mild Stress": states.count("Mild Stress") / max(1, len(states)) * 100,
    "Overload Risk": states.count("Overload Risk") / max(1, len(states)) * 100
}

# Emotion distribution
emotion_counts = {emo: emotions.count(emo) for emo in set(emotions)}
emotional_balance_index = (emotion_counts.get("happy",0) + emotion_counts.get("neutral",0)) / max(1, len(emotions))

# Recovery score approximation
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
recovery_score = 1 - (sum(recovery_intervals)/max(1,len(states)))  # higher better

# Engagement & compliance
engagement_duration = duration
avg_volume = np.mean(volumes) if volumes else 0

# Clinical / Autism-specific: caregiver input
print("\nPlease rate the session (1-5):")
caregiver_ratings = {}
caregiver_ratings['calmness'] = int(input("Calmness: "))
caregiver_ratings['attention'] = int(input("Attention: "))
caregiver_ratings['willingness'] = int(input("Willingness to engage: "))

# Metrics dictionary
metrics = {
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
    "caregiver_ratings": caregiver_ratings
}

# Save metrics
with open(f"session_metrics_{session_id}.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"=== Metrics saved: session_metrics_{session_id}.json ===")

# ---------------------------
# Visualization
# ---------------------------
times = [e['time']-session_start for e in session_log]
plt.figure(figsize=(12,8))

# Stress over time
plt.subplot(3,1,1)
plt.plot(times, stress_values, color='red')
plt.title("Stress Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Stress Score")
plt.grid(True)

# Music volume vs stress
plt.subplot(3,1,2)
plt.plot(times, volumes, label='Music Volume', color='blue')
plt.plot(times, stress_values, label='Stress Score', color='red', alpha=0.5)
plt.title("Music Volume vs Stress")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# Emotion distribution
plt.subplot(3,1,3)
plt.bar(emotion_counts.keys(), emotion_counts.values(), color='green')
plt.title("Emotion Distribution")
plt.xlabel("Emotions")
plt.ylabel("Count")
plt.grid(True)

plt.tight_layout()
plt.savefig(f"session_visuals_{session_id}.png")
plt.show()








