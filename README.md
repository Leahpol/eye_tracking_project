# eye_tracking_project

## Project overview

Real-time eye tracking system that detects and classifies eye states (open/closed/winking) using computer vision.


## Technical Specifications

### Input:

- Video Source: Webcam
- Frame Rate: ≥ 30 FPS
- Resolution: 640×480 or higher

### Output:

- Real-time video feed
- Eye contour visualization (landmark overlays)
- Eye state label (OPEN / CLOSED / WINKING)
- Current EAR value
- Frame counter
- Console output:
    - Detection status
    - Overall blinking frequnecy (blinks per minute) for the stream
    - Duration of the longest blink detected during the strem 


## Algorithm Requirements

### Eye Aspect Ratio (EAR) Formula:

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

where:
- p1, p4 ->  horizontal eye corners (left, right),
- p2, p3, p5, p6 -> vertical eye landmarks (top and bottom),
- || || -> Euclidean distance.

### Classification Threshold:

- EAR < 0.18 → Eyes CLOSED
- EAR ≥ 0.18 → Eyes OPEN
- can be changed with user input 

### MediaPipe Face Mesh Landmarks:

- Left Eye: Indices [362, 385, 387, 263, 373, 380]
- Right Eye: Indices [33, 160, 158, 133, 153, 144]


## Project Structure

    eye_tracking_project/
    ├── eye_tracker.py          # Main implementation
    ├── requirements.txt        # Python dependencies
    ├── README.md               # Project documentation
    └── tests/                  # Functionality and unit tests
        ├── test_ear.py             # Functionality of EAR calculation
        ├── test_extract_eyes.py    # Functionality of Eye landmark extraction
        ├── test_eye_state.py       # Functionality of Eye state Classification 
        ├── test_face_mesh.py       # Functionality of Face detection
        ├── test_webcam.py          # Functionality of Environment set up
        └── u_tests.py              # unit tests


## Installation

1. clone the repository:

```bash
git clone <repository-url>
cd eye_tracking_project
```

2. create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
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

## Testing

```bash
python -m unittest tests/u_tests.py
```

## Known limitations

- Supports webcam video only
- Detects one face 
- Sensitive to lighting conditions
- Winking detection may be inconsistent when not looking stright into the camera




