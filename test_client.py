import cv2
import requests
import time

# URL of your FastAPI endpoint hi
API_URL = "http://0.0.0.0:8000/analyze-posture/"  # Change to your server IP if needed

# Open the webcam (use 0) or a video file path
cap = cv2.VideoCapture(0)  # or cap = cv2.VideoCapture("video.mp4")

frame_interval = 1  # Seconds between frames to avoid overloading the server

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of stream or failed to read frame.")
        break

    # Encode the frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_bytes = img_encoded.tobytes()

    try:
        # Send to FastAPI
        files = {'file': ('frame.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(API_URL, files=files)

        if response.ok:
            print("✅ Response:", response.json())
        else:
            print("❌ Error:", response.status_code, response.text)

    except Exception as e:
        print("⚠️ Exception:", e)

    # Wait between frames (optional)
    time.sleep(frame_interval)

    # Optional: break after a few frames or on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
