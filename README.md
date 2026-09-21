# 🎓 Smart Exam Proctoring System using AI & Computer Vision

## 📌 Project Overview

The **Smart Exam Proctoring System** is an AI-powered computer vision application developed to automatically analyze examination videos and identify potentially suspicious activities.

The system processes an uploaded examination video frame by frame and monitors different events such as:

- Face absence
- Person absence
- Mobile phone usage
- Book presence
- Suspicious events persisting across multiple frames

Instead of relying only on individual object detections, the system combines computer vision, event persistence logic, evidence storage, and a rule-based risk analytics engine to generate an overall examination risk report.

The project supports two examination environments:

- **Online Exam Mode**
- **Classroom Exam Mode**

A Flask-based web application allows the user to upload a video, select the examination mode, run the AI analysis, and view the generated report and supporting evidence.

---

# 🎯 Problem Statement

Traditional examination monitoring requires continuous observation by human invigilators.

In online and large classroom examinations, manually monitoring every student throughout an entire examination can be difficult and time-consuming.

The objective of this project is to develop an automated AI-assisted proctoring pipeline capable of analyzing examination videos and flagging events that may require human review.

The system is designed to:

1. Process examination videos automatically.
2. Detect faces and people.
3. Detect prohibited objects such as mobile phones and books.
4. Reduce repeated alerts using temporal event logic.
5. Capture evidence for detected events.
6. Calculate an overall risk score.
7. Present results through a simple web interface.
8. Handle invalid or corrupt video inputs gracefully.

> **Important:** The system is intended as an AI-assisted monitoring prototype. A detected event does not automatically prove cheating and should be reviewed in context by a human.

---

# 🧠 System Architecture

The overall processing pipeline is:

```text
                    Examination Video
                           │
                           ▼
                    Flask Web Interface
                           │
                           ▼
                    Video Validation
                           │
                           ▼
                    OpenCV Frame Reader
                           │
                           ▼
              ┌─────────────────────────┐
              │    Frame-by-Frame AI    │
              │        Analysis         │
              └─────────────────────────┘
                    │             │
                    ▼             ▼
             Face Detection    YOLOv8 Detection
                    │             │
                    │       ┌─────┴──────┐
                    │       ▼            ▼
                    │    Person       Objects
                    │    Count      Phone / Book
                    │
                    └─────────┬───────────┘
                              ▼
                       Behavior Logic
                              │
                              ▼
                       Evidence Storage
                              │
                              ▼
                       Risk Analytics
                              │
                              ▼
                       Report Generator
                              │
                              ▼
                       Flask Results Page
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Video reading and frame processing |
| YOLOv8 | Person and forbidden-object detection |
| Ultralytics | YOLO model framework |
| Haar Cascade | Face detection |
| Flask | Web application backend |
| HTML | User interface templates |
| Git | Version control |
| GitHub | Source-code hosting and project documentation |

---

# 📁 Project Structure

```text
smart_exam_proctoring/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── object_detector.py
│   ├── object_detector_with_face.py
│   ├── report_generator.py
│   ├── task1_frame_detector.py
│   ├── task2_face_monitoring.py
│   ├── task3_person_count_detector.py
│   ├── task4_forbidden_object_detection.py
│   ├── task5_behavior_logic.py
│   ├── task6_report_generator.py
│   ├── task7_evidence_storage.py
│   ├── task8_risk_analytics.py
│   ├── task9_error_handler.py
│   └── haarcascade_frontalface_default.xml
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── error.html
│
├── docs/
│   └── screenshots/
│       ├── online_exam/
│       ├── classroom_cheating/
│       ├── exam_cheat_phone/
│       └── normal_behavior/
│
└── static/
    ├── uploads/
    └── evidence/
```

Runtime-generated folders, uploaded videos, evidence frames, model weights, and other unnecessary large files are excluded from GitHub using `.gitignore`.

---

# 🔍 Project Development — Task-by-Task Implementation

## Task 1 — Video and Frame Processing

The first stage of the project was to build the video-processing pipeline.

OpenCV is used to open an examination video and read it frame by frame.

Each successfully read frame is passed to the detection pipeline.

Conceptually:

```python
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Analyze frame
```

This converts a complete examination video into individual frames that can be analyzed by the computer vision models.

---

# 👤 Task 2 — Face Monitoring

Face monitoring was implemented to determine whether a face is visible in the examination frame.

The project uses a Haar Cascade face detector.

The face-monitoring module returns the number of detected faces.

If:

```text
face_count = 0
```

the system treats the frame as a possible **face-missing event**.

The purpose of this module is to identify periods during which the candidate may not be visible to the camera.

### Challenge Faced

Initially, the face-violation result frequently remained:

```text
Face Violations: 0
```

even while other detections were working.

This required checking the face-detection return value and the violation condition.

The face logic was updated to explicitly handle both:

```python
face_count is None
```

and:

```python
face_count == 0
```

using logic such as:

```python
if face_count is None or face_count == 0:
    face_violations += 1
```

This made the face-monitoring logic more defensive and easier to debug.

---

# 👥 Task 3 — Person Count Detection

YOLOv8 is used to identify people present in each frame.

The person-count module analyzes the YOLO detections and counts objects classified as `person`.

Person monitoring is especially relevant in **Classroom Mode**.

For example:

```python
if mode == "CLASSROOM" and person_count == 0:
    person_violations += 1
```

This allows the system to apply different rules depending on the selected examination environment.

---

# 📱📚 Task 4 — Forbidden Object Detection

The project uses YOLOv8 to detect objects that may be relevant during an examination.

The primary monitored objects in the current prototype are:

- Mobile phone
- Book

The detector returns Boolean values representing whether these objects were detected in a frame.

Conceptually:

```python
phone_detected, book_detected = detect_forbidden_objects(...)
```

This information is then passed to the behavior-analysis module.

---

# ⏱️ Task 5 — Temporal / Streak-Based Behavior Logic

One of the important improvements made during development was avoiding the treatment of every detected frame as a separate cheating event.

For example, if a phone remains visible for 30 consecutive frames, counting all 30 frames as 30 independent events would exaggerate the result.

To reduce this problem, the project implements **streak-based event detection**.

A detection must persist across a predefined number of frames before being registered as an event.

The project uses:

```python
STREAK_THRESHOLD = 15
```

The system maintains separate streak counters:

```python
phone_streak
book_streak
```

The counters are updated through:

```python
update_streak()
```

This converts repeated frame-level detections into more meaningful event-level information.

### Why This Was Important

Without temporal logic:

```text
1 phone visible for many frames
        ↓
many phone violations
```

With streak logic:

```text
Repeated phone detections
        ↓
Persistent sequence
        ↓
Single meaningful event
```

This reduces unnecessary repeated alerts.

---

# 📄 Task 6 — Automated Report Generation

After the complete video has been processed, the collected information is passed to the report-generation module.

The final report contains values such as:

```text
Exam Mode
Total Frames
Face Violations
Person Violations
Phone Events
Book Events
Risk Score
Risk Percentage
Risk Level
```

Example output:

```text
Exam Analysis Complete

Exam Mode: classroom
Total Frames: 692

Face Violations: 0
Person Violations: 0
Phone Events: 0
Book Events: 6

Risk Score: 120
Risk Percentage: 17.34%
Risk Level: MODERATE
```

The exact result depends on the analyzed video and configured risk rules.

---

# 📸 Task 7 — Evidence Storage

A proctoring report becomes much more useful when suspicious events can be reviewed visually.

Therefore, the project automatically saves selected frames when relevant events occur.

Evidence can be stored for events such as:

```text
phone
book
face_missing
person_missing
```

Example:

```python
path = save_evidence(
    frame,
    "phone",
    total_frames
)
```

The evidence path is then stored so that Flask can display the image on the results page.

This creates a workflow of:

```text
Detection
   ↓
Event confirmation
   ↓
Evidence frame saved
   ↓
Evidence displayed in report
```

This improves traceability because a reviewer can inspect why the system produced an alert.

---

# 📊 Task 8 — Risk Analytics

The system does not simply display raw detection counts.

A separate risk analytics module combines the detected events and produces:

- Risk Score
- Risk Percentage
- Risk Level

The inputs include:

```text
Total Frames
Face Violations
Person Violations
Phone Events
Book Events
```

The risk engine applies predefined rule-based weights to these events.

The final result can classify an examination into levels such as:

```text
LOW
MODERATE
HIGH
```

### Why a Separate Risk Module Was Created

Keeping risk analytics separate from object detection makes the project modular.

The detector answers:

> "What was detected?"

The risk engine answers:

> "How should the configured system summarize those detected events?"

This separation makes future modification of the scoring rules easier.

---

# 🛡️ Task 9 — Error Handling and Robustness

Error handling was added to prevent the Flask application from crashing when invalid input is provided.

The system validates the uploaded video before running the expensive detection pipeline.

The project considers situations such as:

- Empty files
- Corrupt videos
- Unsupported or invalid video content
- Videos that OpenCV cannot open
- Frames that cannot be read

A separate module was created:

```text
src/task9_error_handler.py
```

During testing, a deliberately invalid/corrupt video produced an OpenCV/FFmpeg message similar to:

```text
moov atom not found
```

Instead of allowing the application to fail with an unhandled exception, the application returns a user-friendly error message such as:

```text
Error Occurred
Cannot open video file. Possibly corrupt.
```

A dedicated Flask template was therefore created:

```text
templates/error.html
```

### Challenge Faced

During the first error-handling test, Flask produced:

```text
jinja2.exceptions.TemplateNotFound: error.html
```

The backend error detection was actually working, but the application did not yet have the HTML page required to display the error.

### Solution

The missing file was created:

```text
templates/error.html
```

After adding the template, corrupt-video errors could be handled without displaying the Flask traceback to the end user.

---

# 🌐 Task 10 — Flask Web Application

The detection pipeline was integrated with Flask to provide a simple web interface.

The user can:

1. Open the application.
2. Select an examination mode.
3. Upload an examination video.
4. Start analysis.
5. Wait while the video is processed.
6. View the generated risk report.
7. Review captured evidence.

The main Flask application is:

```text
app.py
```

The main HTML templates are:

```text
templates/index.html
templates/result.html
templates/error.html
```

This transformed the project from individual computer-vision scripts into an end-to-end application.

---

# 🖥️ Examination Modes

## Online Exam Mode

Online Exam Mode is designed for webcam-style individual examination scenarios.

The system can monitor signals such as:

- Candidate face visibility
- Phone detection
- Book detection
- Persistent suspicious-object events

## Classroom Exam Mode

Classroom Mode allows the project to process examination videos containing classroom environments.

The mode includes person-related monitoring in addition to forbidden-object detection.

Mode-specific logic prevents every examination environment from being treated identically.

---

# 📊 Example Analysis Results

One classroom test produced:

```text
Exam Mode: classroom
Total Frames: 692
Face Violations: 0
Person Violations: 0
Phone Events: 0
Book Events: 6
Risk Score: 120
Risk Percentage: 17.34%
Risk Level: MODERATE
```

Another online examination test produced:

```text
Exam Mode: online
Total Frames: 639
Face Violations: 0
Person Violations: 0
Phone Events: 2
Book Events: 0
Risk Score: 100
Risk Percentage: 15.65%
Risk Level: HIGH
```

These examples also illustrate that the final risk level is determined by the configured risk rules and event weights rather than simply by the number of events.

---

# 📸 Application Screenshots

## Online Examination

![Online Exam 1](docs/screenshots/online_exam/online_exam_1.png)

![Online Exam 2](docs/screenshots/online_exam/online_exam_2.png)

![Online Exam 3](docs/screenshots/online_exam/online_exam_3.png)

---

## Classroom Cheating Analysis

![Classroom Cheating 1](docs/screenshots/classroom_cheating/classroom_cheating_1.png)

![Classroom Cheating 2](docs/screenshots/classroom_cheating/classroom_cheating_2.png)

![Classroom Cheating 3](docs/screenshots/classroom_cheating/classroom_cheating_3.png)

---

## Mobile Phone Detection

![Phone Detection 1](docs/screenshots/exam_cheat_phone/exam_cheat_phone_1.png)

![Phone Detection 2](docs/screenshots/exam_cheat_phone/exam_cheat_phone_2.png)

![Phone Detection 3](docs/screenshots/exam_cheat_phone/exam_cheat_phone_3.png)

---

## Normal Behavior

![Normal Behavior 1](docs/screenshots/normal_behavior/normal_behavior_1.png)

![Normal Behavior 2](docs/screenshots/normal_behavior/normal_behavior_2.png)

![Normal Behavior 3](docs/screenshots/normal_behavior/normal_behavior_3.png)

---

# 🚧 Major Challenges Faced and Solutions

## 1. Repeated Object Detection

### Problem

YOLO detects objects frame by frame.

A phone visible for several consecutive frames could therefore be counted repeatedly.

### Solution

A streak-based temporal logic was introduced.

Only detections persisting for the configured threshold are converted into events.

---

## 2. Face Violation Logic

### Problem

Face violations were sometimes reported as zero even when the face detector needed additional defensive handling.

### Solution

The face-count result was explicitly checked for both `None` and zero.

This improved the robustness of the face-monitoring condition.

---

## 3. Corrupt Video Handling

### Problem

An invalid video could not be opened by OpenCV and produced errors such as:

```text
moov atom not found
```

### Solution

A dedicated video-validation/error-handling module was added before full inference.

Invalid input now produces a controlled error response.

---

## 4. Missing Error Template

### Problem

After backend error handling was implemented, Flask generated:

```text
TemplateNotFound: error.html
```

### Solution

A dedicated:

```text
templates/error.html
```

page was created.

This allowed the application to show readable error messages instead of a development traceback.

---

## 5. Evidence Management

### Problem

Saving every detection would create hundreds or thousands of unnecessary images.

### Solution

Evidence storage was connected to event logic so that selected event frames could be saved instead of treating every frame as independent evidence.

---

## 6. Large Number of Runtime Files

### Problem

Testing generated many:

- Evidence images
- Uploaded videos
- Output files
- Temporary files
- Model files

These should not unnecessarily increase the size of the GitHub repository.

### Solution

A `.gitignore` file was created to exclude runtime-generated and large local files.

Examples include:

```text
__pycache__/
*.pyc
venv/
*.pt
uploads/
outputs/
data/
static/uploads/
static/evidence/
```

Project demonstration screenshots are intentionally stored separately under:

```text
docs/screenshots/
```

so that they can still be displayed in the README.

---

## 7. Python Cache Files

Python automatically creates:

```text
__pycache__/
```

containing compiled `.pyc` files.

These are machine-generated files and are not required in the source repository.

They are therefore excluded using `.gitignore`.

---

## 8. Organizing Documentation Images

Initially, screenshot folders and image files contained spaces and inconsistent capitalization.

For example:

```text
Images of Online_exam
images of Normal Behavior
```

They were reorganized into a cleaner structure:

```text
docs/screenshots/
    classroom_cheating/
    exam_cheat_phone/
    normal_behavior/
    online_exam/
```

Image filenames were also standardized:

```text
online_exam_1.png
normal_behavior_1.png
classroom_cheating_1.png
exam_cheat_phone_1.png
```

This makes README image paths simpler and the repository easier to maintain.

---

# ⚠️ Current Limitations

The current project is a prototype and has several limitations.

### Face Detection

Haar Cascade detection can be affected by:

- Low lighting
- Face angle
- Occlusion
- Camera quality
- Distance from the camera

### Object Detection

YOLO detections depend on:

- Image quality
- Object size
- Camera angle
- Occlusion
- Confidence threshold
- Classes supported by the pretrained model

### Rule-Based Risk Scoring

The current risk score is based on predefined rules and weights.

It is not a statistically calibrated probability that cheating occurred.

### Behavior Understanding

The system detects observable visual events, but it cannot fully understand a student's intent.

For example, the presence of a book or phone does not by itself establish misconduct.

### Generalization

Performance can vary across different:

- Examination rooms
- Camera positions
- Lighting conditions
- Video resolutions
- Number of students
- Device types

Therefore, human review remains important.

---

# 🚀 Future Improvements

Future versions of the system could include:

- Deep-learning-based face detection instead of Haar Cascade
- Face tracking across frames
- Multiple-person tracking
- Head-pose estimation
- Gaze estimation
- Audio monitoring
- Improved temporal behavior modeling
- Custom YOLO model training on exam-specific data
- Better low-light processing
- Confidence-based event filtering
- Database integration
- User authentication
- Admin dashboard
- Real-time webcam monitoring
- Live alerts
- PDF report generation
- Cloud deployment
- Docker support
- Automated testing
- Configurable risk weights
- Better risk visualization
- Model performance benchmarking

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
```

Move into the project:

```bash
cd smart_exam_proctoring
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

Git Bash:

```bash
source venv/Scripts/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Model Weights

Large `.pt` model files are excluded from this repository through `.gitignore`.

The required YOLO weights should therefore be downloaded/generated separately and placed in the location expected by the application.

---

## 5. Run the Application

```bash
python app.py
```

Flask will start the local development server.

Open the address displayed in the terminal, typically:

```text
http://127.0.0.1:5000
```

---

# 🧪 Testing the Application

The application was tested using multiple types of examination videos, including:

- Normal examination behavior
- Online examination scenarios
- Classroom examination scenarios
- Phone-related scenarios
- Book-related scenarios
- Invalid/corrupt video input

Testing different scenarios helped verify both the detection pipeline and the application's error-handling behavior.

---

# 📈 What I Learned From This Project

This project provided practical experience in building an end-to-end computer vision application rather than only running a machine-learning model.

Major concepts learned include:

- Video processing with OpenCV
- Frame-by-frame inference
- YOLOv8 object detection
- Face detection
- Person counting
- Forbidden-object detection
- Temporal event logic
- Evidence management
- Rule-based risk analytics
- Flask backend development
- HTML template integration
- Error handling
- Debugging Python imports
- Organizing modular Python code
- Git and GitHub version control
- `.gitignore` management
- Technical documentation

One of the most important lessons from the project was that an AI application requires more than a detection model.

A complete system also requires:

```text
Input validation
        +
Computer vision
        +
Behavior logic
        +
Evidence management
        +
Risk analytics
        +
Error handling
        +
User interface
        +
Documentation
```

---

# 🔐 Responsible Use

Automated proctoring can produce false positives and may be affected by environmental and technical conditions.

This project should therefore be considered an **AI-assisted review system**, not an autonomous system for determining whether a student cheated.

Any consequential decision should involve appropriate human review and consideration of the original examination context.

---

# 📌 Conclusion

The Smart Exam Proctoring System demonstrates how computer vision models can be integrated into a complete AI application.

The project combines:

**OpenCV + YOLOv8 + Face Detection + Temporal Event Logic + Evidence Capture + Risk Analytics + Flask**

to process examination videos and generate structured monitoring reports.

The modular architecture also makes it possible to independently improve the detection models, behavior logic, risk engine, web interface, and reporting components in future versions.
