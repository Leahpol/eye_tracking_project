# eye_tracking_project

## Project overview

Real-time eye tracking system that detects and classifies eye states (open/closed/winking) using computer vision.

### Features:

- Real time eye state detection with contour overlay (Open: green, Closed: red)
- Winking detection (per eye state tracking)
- Overall blinking frequnecy (blinks per minute) for the stream
- Duration of the longest blink detected during the strem 

## Technical Specifications

### Input:

- Video Source: Webcam
- Frame Rate: ≥ 30 FPS
- Resolution: 640×480 or higher

### Output:

- Real-time video feed
- Eye contour visualization (landmark overlays)
- Eye state label (OPEN / CLOSED)
- Current EAR value
- Frame counter
- Console output indicating detection status


## Algorithm Requirements

### Eye Aspect Ratio (EAR) Formula:

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

where:
- p1, p4 ->  horizontal eye corners (left, right),
- p2, p3, p5, p6 -> vertical eye landmarks (top and bottom),
- || || -> Euclidean distance.

### Classification Threshold:

- EAR < 0.21 → Eyes CLOSED
- EAR ≥ 0.21 → Eyes OPEN

### MediaPipe Face Mesh Landmarks:

- Left Eye: Indices [362, 385, 387, 263, 373, 380]
- Right Eye: Indices [33, 160, 158, 133, 153, 144]


## Project Structure

    eye_tracking_project/
    ├── eye_tracker.py          # Main implementation
    ├── requirements.txt        # Python dependencies
    ├── README.md               # Project documentation
    └── tests/                  # tasks 1-5 independently 
        ├── test_ear.py             # Test EAR calculation
        ├── test_extract_eyes.py    # Eye landmark extraction
        ├── test_eye_state.py       # Eye state Classification 
        ├── test_face_mesh.py       # Face detection
        └── test_webcam.py          # Environment set up


## Installation

1. clone the repository:

```bash
git clone <repository-url>
cd eye_tracking_project
```

2. create and activate a virtual environment:

```bash
python3 -m venv venv
source .venv/bin/activate
```

3. Install required Dependencies:

```bash
pip install opencv-python mediapipe numpy scipy
```

## Usage examples

```bash
python eye_tracker.py
```

- Press 'q' to stop the tracker 


## Known limitations

- Supports webcam video only
- Sensitive to lighting conditions
- Winking detection may be inconsistent 




