# Posture Analysis System

A real-time posture analysis system built with FastAPI and MediaPipe that analyzes body posture from images and provides feedback on shoulder, head, and spine alignment.

## Features

- **Real-time Posture Analysis**: Upload images to get instant posture feedback
- **Multiple Metrics**: Analyzes shoulder tilt, head tilt, spine alignment, and side tilts
- **Visual Feedback**: Returns annotated images with pose landmarks
- **Remote Control Mode**: Toggle forced responses for testing/demo purposes
- **RESTful API**: Easy integration with any application
- **Webcam Integration**: Test client for real-time webcam analysis

## System Architecture

- **main.py**: FastAPI server with posture analysis endpoints
- **mode.py**: Remote control utility to toggle forced response mode
- **test_client.py**: Webcam test client for real-time analysis

## Installation

### Prerequisites

```bash
pip install fastapi uvicorn opencv-python mediapipe numpy pynput requests
```

### Dependencies

- **FastAPI**: Web framework for the API
- **OpenCV**: Image processing
- **MediaPipe**: Pose detection and landmark extraction
- **NumPy**: Numerical computations
- **Uvicorn**: ASGI server
- **Pynput**: Keyboard input handling (for mode control)
- **Requests**: HTTP client (for test client)

## Usage

### 1. Start the Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`

### 2. Analyze Posture

Send a POST request to `/analyze-posture/` with an image file:

```bash
curl -X POST "http://localhost:8000/analyze-posture/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

### 3. Remote Mode Control

Run the mode control utility:

```bash
python mode.py
```

- Press `t` to toggle forced response mode ON/OFF
- Press `q` to quit

### 4. Real-time Webcam Testing

```bash
python test_client.py
```

- Uses your default webcam (camera 0)
- Sends frames to the API every second
- Press `q` to quit

## API Endpoints

### POST `/analyze-posture/`

Analyzes posture from an uploaded image.

**Parameters:**
- `file`: Image file (JPG, PNG, etc.)

**Response:**
```json
{
  "posture_score": 95.67,
  "posture_check": "Posture Accepted",
  "tilt_angles": {
    "shoulder_tilt": {
      "angle": 0.5,
      "score": 98.5,
      "feedback": "Shoulder Good",
      "bool": true
    },
    "head_tilt": {
      "angle": 0.3,
      "score": 99.2,
      "feedback": "Head Good",
      "bool": true
    },
    "spine_tilt": {
      "angle": 2.1,
      "score": 97.9,
      "feedback": "Spine Good",
      "bool": true
    },
    "left_tilt": {
      "angle": 1.2,
      "feedback": "Good"
    },
    "right_tilt": {
      "angle": 1.5,
      "feedback": "Good"
    }
  },
  "img_base64": "base64_encoded_annotated_image"
}
```

### POST `/set-mode/`

Controls the forced response mode for testing purposes.

**Parameters:**
- `enabled`: Boolean to enable/disable forced mode
- `message`: Optional message for forced responses

## Posture Metrics

### Scoring System

- **Posture Score**: Overall score (0-100) averaged from all metrics
- **Acceptance Threshold**: Score ≥ 98 for "Posture Accepted"
- **Individual Scores**: Based on deviation from ideal alignment

### Measured Angles

1. **Shoulder Tilt**: Angle between left and right shoulders
2. **Head Tilt**: Angle between left and right ears
3. **Spine Tilt**: Angle between shoulder and hip alignment
4. **Side Tilts**: Left and right ear-to-shoulder angles

### Thresholds

- **Shoulder/Head Tilt**: ≤ 0.7° for "Good"
- **Spine Tilt**: ≤ 90° for "Good"
- **Side Tilt**: ≤ 3° for "Good"

## Configuration

### Server Configuration

Default server runs on:
- **Host**: `0.0.0.0`
- **Port**: `8000`
- **Reload**: Enabled for development

### MediaPipe Settings

```python
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

## Error Handling

- **No Person Detected**: Returns 400 error if no pose landmarks found
- **Invalid Image**: Handles corrupted or invalid image files
- **Network Errors**: Test client includes exception handling

## Development

### Running in Development Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Testing

Use the provided test client or any HTTP client to test the API:

```python
import requests

files = {'file': open('test_image.jpg', 'rb')}
response = requests.post('http://localhost:8000/analyze-posture/', files=files)
print(response.json())
```

## Production Deployment

For production deployment:

1. Use a production ASGI server like Gunicorn
2. Configure proper logging
3. Add authentication if needed
4. Set up HTTPS
5. Configure CORS if serving web clients

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Troubleshooting

### Common Issues

1. **Camera not found**: Check camera index in test_client.py
2. **No pose detected**: Ensure good lighting and full body visibility
3. **Connection errors**: Verify server URL and port
4. **Permission errors**: Check camera permissions for webcam access

### Performance Tips

- Adjust `frame_interval` in test client to reduce server load
- Use lower resolution images for faster processing
- Consider caching for repeated analysis of similar poses

## License

This project is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!
