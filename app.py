from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
from queue import Queue
from datetime import datetime
import time
import atexit
import logging

# ------------------ Logging Setup ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ------------------ TTS Setup ------------------
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('voice', 'english')

speech_queue = Queue()
stop_signal = threading.Event()

def speech_worker():
    while not stop_signal.is_set():
        if not speech_queue.empty():
            text = speech_queue.get()
            try:
                logger.info(f"TTS speaking: {text}")
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"Error in TTS: {e}")
        time.sleep(0.1)

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# ------------------ Emotion Detection Setup ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

emotion_config = {
    "happy": {"color": (0, 255, 0), "message": "You look happy! Keep smiling!"},
    "sad": {"color": (255, 0, 0), "message": "You seem sad. I hope you feel better soon."},
    "angry": {"color": (0, 0, 255), "message": "You appear angry. Take a deep breath."},
    "surprise": {"color": (0, 255, 255), "message": "Wow! You look surprised!"},
    "neutral": {"color": (200, 200, 200), "message": "Your expression is neutral."},
    "fear": {"color": (255, 140, 0), "message": "You look fearful. Everything's okay."},
    "disgust": {"color": (138, 43, 226), "message": "You're showing signs of disgust."}
}

emotion_history = {
    "timestamps": [],
    "emotions": [],
    "current_streak": {"emotion": None, "count": 0}
}

def distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def get_emotion(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    iris_left = landmarks[468]

    face_width = distance(landmarks[234], landmarks[454])
    mouth_open = distance(top_lip, bottom_lip) / face_width
    mouth_stretch = distance(left_mouth, right_mouth) / face_width
    eye_open = (distance(left_eye_top, left_eye_bottom) + distance(right_eye_top, right_eye_bottom)) / (2 * face_width)
    eye_center_y = (left_eye_top.y + left_eye_bottom.y + right_eye_top.y + right_eye_bottom.y) / 4
    sad_offset = iris_left.y - eye_center_y

    if mouth_stretch > 0.40 and mouth_open < 0.06:
        return "happy"
    elif mouth_open >= 0.12:
        return "surprise"
    elif 0.06 < mouth_open < 0.12:
        return "fear"
    elif sad_offset > 0.01 and eye_open < 0.04:
        return "sad"
    elif mouth_open < 0.03 and eye_open < 0.08 and mouth_stretch < 0.38:
        return "disgust"
    elif eye_open > 0.096 and mouth_open < 0.06:
        return "angry"
    else:
        return "neutral"

def update_emotion_history(emotion):
    now = datetime.now()
    emotion_history["timestamps"].append(now.strftime("%H:%M:%S"))
    emotion_history["emotions"].append(emotion)

    if len(emotion_history["timestamps"]) > 50:
        emotion_history["timestamps"] = emotion_history["timestamps"][-50:]
        emotion_history["emotions"] = emotion_history["emotions"][-50:]

    if emotion_history["current_streak"]["emotion"] == emotion:
        emotion_history["current_streak"]["count"] += 1
    else:
        emotion_history["current_streak"] = {"emotion": emotion, "count": 1}

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_emotion = None
    last_spoken_time = 0

    if not cap.isOpened():
        logger.error("Webcam not accessible.")
        return

    logger.info("Starting webcam stream...")

    while True:
        success, frame = cap.read()
        if not success:
            logger.error("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        emotion = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmark_list = face_landmarks.landmark
            emotion = get_emotion(landmark_list)
            update_emotion_history(emotion)

            color = emotion_config.get(emotion, {}).get("color", (255, 255, 255))
            cv2.putText(frame, f"Emotion: {emotion}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            current_time = time.time()
            if emotion != prev_emotion and (current_time - last_spoken_time) > 5:
                message = emotion_config.get(emotion, {}).get("message", "")
                if message and not stop_signal.is_set():
                    logger.info(f"Queued message for TTS: {message}")
                    speech_queue.put(message)
                    last_spoken_time = current_time
                prev_emotion = emotion
        else:
            prev_emotion = None

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion_data')
def get_emotion_data():
    emotion_counts = {e: emotion_history["emotions"].count(e) for e in set(emotion_history["emotions"])}
    return jsonify({
        "current_emotion": emotion_history["emotions"][-1] if emotion_history["emotions"] else None,
        "emotion_counts": emotion_counts,
        "current_streak": emotion_history["current_streak"],
        "history": list(zip(emotion_history["timestamps"], emotion_history["emotions"]))[-10:]
    })

@app.route('/toggle_voice', methods=['POST'])
def toggle_voice():
    if stop_signal.is_set():
        stop_signal.clear()
        logger.info("Voice feedback enabled.")
        return jsonify({"status": "success", "voice_enabled": True})
    else:
        stop_signal.set()
        logger.info("Voice feedback disabled.")
        return jsonify({"status": "success", "voice_enabled": False})

def on_shutdown():
    stop_signal.set()
    engine.stop()
    logger.info("Application shutdown and TTS engine stopped.")

atexit.register(on_shutdown)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
