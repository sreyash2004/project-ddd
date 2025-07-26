from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pyttsx3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading

app = Flask(__name__)

# Global variables
cap = None
detection_active = False
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define landmark indices for eyes
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Counter to track consecutive drowsy frames
drowsy_counter = 0
warning_counter = 0
email_sent = False  # Prevent multiple email alerts

# Email Configuration
EMAIL_ADDRESS = "saniagemmathew@gmail.com"  # Your email
EMAIL_PASSWORD = "ndwtuhlkvyjgcqqv"  # Your email app password
RELATIVE_EMAIL = "shsreya2@gmail.com"  # Relative's email address


def compute(ptA, ptB):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(ptA - ptB)


def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) to detect blinks."""
    A = compute(eye[1], eye[5])  # Vertical left
    B = compute(eye[2], eye[4])  # Vertical right
    C = compute(eye[0], eye[3])  # Horizontal

    ear = (A + B) / (2.0 * C)
    return ear


def blinked(ear):
    """Check if eyes are closed or blinking based on EAR."""
    if ear < 0.21:
        return 0  # Eyes closed
    elif 0.21 <= ear <= 0.25:
        return 1  # Drowsy
    else:
        return 2  # Eyes open


def play_alert(frame):
    """Play audio alert using text-to-speech when drowsiness is detected."""
    global email_sent

    print("🔊 Playing alert...")

    # Start audio alert
    engine = pyttsx3.init()
    alert_message = "Warning! You are feeling drowsy. Please take a break."
    engine.say(alert_message)
    engine.runAndWait()

    # Send email with frame only if not already sent
    if not email_sent:
        send_email_alert_threaded(frame)
        email_sent = True  # Set flag after sending email


def send_email_alert(frame):
    """Send an email alert with the drowsy frame attached."""
    try:
        print("📧 Sending Email alert to relative with frame...")

        # Save the drowsy frame as an image
        frame_path = "drowsy_frame.jpg"
        cv2.imwrite(frame_path, frame)  # Save current frame

        # Email content
        subject = "URGENT ALERT: Driver is Drowsy!"
        body = "URGENT ALERT! The driver is feeling drowsy. Please take immediate action."

        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RELATIVE_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Attach the image
        with open(frame_path, 'rb') as img_file:
            img = MIMEImage(img_file.read(), name="drowsy_frame.jpg")
            msg.attach(img)

        # SMTP connection and sending email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RELATIVE_EMAIL, msg.as_string())
        server.quit()
        print("✅ Email Alert with Frame Sent Successfully!")
    except smtplib.SMTPAuthenticationError as auth_err:
        print(f"❌ Authentication Error: {auth_err}")
    except smtplib.SMTPException as smtp_err:
        print(f"❌ SMTP Error: {smtp_err}")
    except Exception as e:
        print(f"❌ General Error: {e}")


def send_email_alert_threaded(frame):
    """Run email alert in a separate thread."""
    threading.Thread(target=send_email_alert, args=(frame,)).start()


def gen_frames():
    """Generate frames from the camera."""
    global cap, detection_active, drowsy_counter, warning_counter, email_sent

    print("✅ Starting video feed...")
    cap = cv2.VideoCapture(0)  # Open camera

    if not cap.isOpened():
        print("❌ Camera not available. Check permissions.")
        return

    drowsy_counter = 0
    warning_counter = 0
    email_sent = False

    while detection_active:
        success, frame = cap.read()

        if not success or frame is None:
            print("❌ No frame captured. Retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[LEFT_EYE]
            right_eye = shape[RIGHT_EYE]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            status = blinked(avg_ear)

            # Draw eye landmarks
            for (x, y) in np.concatenate([left_eye, right_eye]):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if status == 0:
                drowsy_counter += 1
                cv2.putText(frame, "😴 ALERT: Eyes Closed!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Trigger alert after 3 seconds (~20 frames at ~7 fps)
                if drowsy_counter >= 20:
                    print("⚠️ Drowsiness Detected! Triggering Alert!")
                    cv2.putText(frame, "⚠️ DROWSY! WAKE UP!", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    # Trigger audio alert and email with frame
                    play_alert(frame)

                    # Reset drowsy counter
                    drowsy_counter = 0

            elif status == 1:
                cv2.putText(frame, "😐 Drowsy", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                drowsy_counter = 0  # Reset when eyes are partially open
            else:
                cv2.putText(frame, "🙂 Awake", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                drowsy_counter = 0  # Reset when eyes are open
                warning_counter = 0  # Reset warnings when awake
                email_sent = False  # Reset email flag when awake

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ Frame encoding error.")
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    print("📴 Camera released after stopping detection.")


@app.route('/')
def index():
    """Render the main page."""
    return open("index.html").read()



@app.route('/video_feed')
def video_feed():
    """Start streaming the video feed."""
    global detection_active

    if detection_active:
        print("🎥 Video feed streaming...")
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        print("⚠️ Detection not active. Start detection first.")
        return "⚠️ Detection not started. Press 'Start Detection'."


@app.route('/start_detection', methods=['POST'])
def start_detection():
    """Start the detection."""
    global detection_active
    if not detection_active:
        print("▶️ Detection started! detection_active = True")
        detection_active = True
    return '', 200


@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    """Stop the detection."""
    global detection_active
    if detection_active:
        print("⏹️ Stopping detection and releasing camera...")
        detection_active = False
    return '', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
