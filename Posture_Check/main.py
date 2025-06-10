from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import math
from typing import Dict
import base64

app = FastAPI()

# Global forced response state
forced_response = {"enabled": False, "response": None}

@app.post("/set-mode/")
async def set_mode(enabled: bool = Form(...), message: str = Form(None)):
    """
    Remotely enable/disable forced response for all API clients.
    """
    forced_response["enabled"] = enabled
    forced_response["response"] = {"message": message} if message else None
    return {"status": "Forced response updated", "current_state": forced_response}

# Setup Mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

@app.post("/analyze-posture/")
async def analyze_posture(file: UploadFile = File(...)) -> Dict[str, object]:
    """
    Analyzes the posture in an uploaded image file and calculates posture scores.

    Args:
        file (UploadFile): The uploaded image file to analyze.

    Returns:
        Dict[str, object]: A dictionary containing the posture score, posture check status,
                           and tilt angles for shoulders, head, and spine.
    """
    
    content = await file.read()
    npimg = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)

    if not results.pose_landmarks:
        return JSONResponse(content={"error": "No person detected"}, status_code=400)

    height, width, _ = frame.shape
    lms = results.pose_landmarks.landmark

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Encode frame to JPEG, then to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    def get_point(lm) -> tuple[int, int, float]:
        """Returns the 2D or 3D coordinates of a landmark."""
        return (
            int(lm.x * width),
            int(lm.y * height),
            lm.z if hasattr(lm, 'z') else 0
        )

    def angle(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Returns the angle between two points."""
        return 180 - abs(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))

    def side_tilt(ear: tuple[float, float, float], shoulder: tuple[float, float, float]) -> float:
        """Calculates the tilt angle of the head relative to the shoulder."""
        dy, dz = abs(ear[1] - shoulder[1]), abs(ear[2] - shoulder[2]) * 100
        return 90 - math.degrees(math.atan2(dy, dz))

    ls = get_point(lms[mp_holistic.PoseLandmark.LEFT_SHOULDER])
    rs = get_point(lms[mp_holistic.PoseLandmark.RIGHT_SHOULDER])
    le = get_point(lms[mp_holistic.PoseLandmark.LEFT_EAR])
    re = get_point(lms[mp_holistic.PoseLandmark.RIGHT_EAR])
    lh = get_point(lms[mp_holistic.PoseLandmark.LEFT_HIP])
    rh = get_point(lms[mp_holistic.PoseLandmark.RIGHT_HIP])

    shoulder_tilt = angle(ls, rs)
    head_tilt = angle(le, re)
    spine_tilt = angle(ls, lh)

    left_tilt = side_tilt(le, ls)
    right_tilt = side_tilt(re, rs)

    shoulder_score = 100 - abs(shoulder_tilt)
    head_score = 100 - abs(head_tilt)
    spine_score = 100 - abs(spine_tilt)
    
    tilt_angles = {
        'shoulder_tilt': shoulder_tilt,
        'head_tilt': head_tilt,
        'spine_tilt': spine_tilt,
    }

    max_tilt = {
        'shoulder_tilt': 0.7,
        'head_tilt': 0.7,
        'spine_tilt': 90,
    }

    def get_score(tilt: float, max_allowed_tilt: float) -> float:
        """Calculates the score based on tilt angle and max allowed tilt."""
        score = 100 - (abs(tilt) / max_allowed_tilt) 
        return max(0, round(score, 2))

    def get_feedback(tilt: float, max_allowed_tilt: float) -> str:
        """Returns feedback based on tilt angle and max allowed tilt."""
        return "Good" if abs(tilt) <= max_allowed_tilt else "Needs Improvement"

    # Calculate scores
    shoulder_score = get_score(shoulder_tilt, max_tilt['shoulder_tilt'])
    head_score = get_score(head_tilt, max_tilt['head_tilt'])
    spine_score = get_score(spine_tilt, max_tilt['spine_tilt'])

    # Compute posture score and check
    posture_score = round((shoulder_score + head_score + spine_score) / 3, 2)
    posture_accepted = posture_score >= 98 
    posture_check = 'Posture Accepted' if posture_accepted else 'Posture Rejected'

    # Override all feedbacks to "Good" if posture is accepted
    def conditional_feedback(tilt: float, max_allowed_tilt: float) -> str:
        """Returns feedback based on tilt angle and max allowed tilt."""
        return "Good" if posture_accepted else get_feedback(tilt, max_allowed_tilt)

    side_feedback = "Good" if posture_accepted else ("Good" if left_tilt <= 3 else "Needs Improvement")

    if forced_response["enabled"]:
        return {
        "posture_score": posture_score,
        "posture_check": 'Posture Accepted',
        "tilt_angles": {
            "shoulder_tilt": {
                "angle": shoulder_tilt,
                "score": shoulder_score,
                "feedback": 'Shoulder '+conditional_feedback(shoulder_tilt, max_tilt['shoulder_tilt']),
                "bool": True,
            },
            "head_tilt": {
                "angle": head_tilt,
                "score": head_score,
                "feedback": 'Head '+conditional_feedback(head_tilt, max_tilt['head_tilt']),
                "bool": True,
            },
            "spine_tilt": {
                "angle": spine_tilt,
                "score": spine_score,
                "feedback": 'Spine '+conditional_feedback(spine_tilt, max_tilt['spine_tilt']),
                "bool": True,
            },
            "left_tilt": {
                "angle": left_tilt,
                "feedback": "Good" if posture_accepted else ("Good" if left_tilt <= 3 else "Needs Improvement")
            },
            "right_tilt": {
                "angle": right_tilt,
                "feedback": "Good" if posture_accepted else ("Good" if right_tilt <= 3 else "Needs Improvement")
            },
        },
        "img_base64": encoded_img,
    }

    return {
        "posture_score": posture_score,
        "posture_check": posture_check,
        "tilt_angles": {
            "shoulder_tilt": {
                "angle": shoulder_tilt,
                "score": shoulder_score,
                "feedback": 'Shoulder '+conditional_feedback(shoulder_tilt, max_tilt['shoulder_tilt']),
                "bool": True if conditional_feedback(shoulder_tilt, max_tilt['shoulder_tilt']) == 'Good' else False,
            },
            "head_tilt": {
                "angle": head_tilt,
                "score": head_score,
                "feedback": 'Head '+conditional_feedback(head_tilt, max_tilt['head_tilt']),
                "bool": True if conditional_feedback(head_tilt, max_tilt['head_tilt']) == 'Good' else False,
            },
            "spine_tilt": {
                "angle": spine_tilt,
                "score": spine_score,
                "feedback": 'Spine '+conditional_feedback(spine_tilt, max_tilt['spine_tilt']),
                "bool": True if conditional_feedback(spine_tilt, max_tilt['spine_tilt']) == 'Good' else False,
            },
            "left_tilt": {
                "angle": left_tilt,
                "feedback": "Good" if posture_accepted else ("Good" if left_tilt <= 3 else "Needs Improvement")
            },
            "right_tilt": {
                "angle": right_tilt,
                "feedback": "Good" if posture_accepted else ("Good" if right_tilt <= 3 else "Needs Improvement")
            },
        },
        "img_base64": encoded_img,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    


