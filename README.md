# AI-Based Secure Face Recognition System

This project is a modular Flask application that implements a secure face authentication workflow using:

- Principal Component Analysis (PCA / Eigenfaces)
- OpenCV face detection
- Geometric eye-distance matching
- SQLite storage
- Admin-controlled user management

The application is organized around three main modules:

- `admin_routes.py`: secure admin portal, training, and log monitoring
- `user_routes.py`: add, update, and delete registered users
- `recognition_routes.py`: public real-time face authentication screen

## Key Features

- Separate admin portal with first-time admin account setup
- Admin login / logout
- User management with `Name`, `User ID`, and `Email`
- Webcam capture of `10-20` face samples per user
- Preprocessing with grayscale conversion, CLAHE, histogram equalization, and normalization
- Processed sample storage in SQLite as compressed feature vectors instead of raw images
- PCA training with strict threshold-based decision logic
- Inter-ocular distance matching using MediaPipe facial landmarks when available, with OpenCV eye-detection fallback
- Authentication logs for granted, denied, no-face, and admin-login events
- Real-time authentication UI that shows user details only when access is granted

## Project Structure

```text
AI-Based Secure Face Recognition System/
|-- app.py
|-- admin_routes.py
|-- auth_utils.py
|-- db.py
|-- face_utils.py
|-- recognition_routes.py
|-- train.py
|-- user_routes.py
|-- requirements.txt
|-- README.md
|-- dataset/
|   `-- captures/
|-- instance/
|   `-- face_recognition.db
|-- models/
|   `-- pca_model.pkl
|-- static/
|   |-- css/
|   |   `-- styles.css
|   `-- js/
|       |-- admin_capture.js
|       `-- authenticate.js
`-- templates/
    |-- base.html
    |-- admin_dashboard.html
    |-- admin_login.html
    |-- admin_logs.html
    |-- admin_setup.html
    |-- admin_user_form.html
    |-- admin_users.html
    `-- authenticate.html
```

## Database Design

The SQLite database file is created at:

- `instance/face_recognition.db`

Main tables:

- `admins`: admin usernames and password hashes
- `users`: registered user details and trained summary data
- `face_samples`: processed compressed feature vectors and average eye-distance values
- `auth_logs`: authentication attempts and monitoring data

## How the System Works

### 1. Admin setup and login

- On first launch, the app redirects to `/admin/setup`
- The first admin account is created securely with a hashed password
- After setup, admins log in through `/admin/login`

### 2. User enrollment

- Admin opens `Users -> Add New User`
- Enters:
  - full name
  - user ID
  - email
- Captures `10-20` webcam samples
- The backend:
  - detects the largest face
  - preprocesses the face image
  - computes normalized eye distance
  - stores compressed processed vectors in SQLite

### 3. PCA training

- Admin clicks `Train / Refresh Model`
- All processed samples are loaded from SQLite
- PCA is fitted to generate Eigenface-style projections
- Mean PCA profiles are produced for each user
- Strict thresholds are derived from the training distribution
- The trained model is saved to:
  - `models/pca_model.pkl`

### 4. Real-time authentication

- Public authentication runs at `/authenticate`
- A live webcam frame is captured
- Face preprocessing and feature extraction are repeated
- The system compares the live face against stored user profiles using:
  - PCA Euclidean distance
  - normalized eye-distance difference
- Decision rule:

```text
If PCA distance < threshold1 AND eye distance difference < threshold2:
    Access Granted
Else:
    Access Denied
```

### 5. Output behavior

- If match succeeds:
  - `Face Detected - Access Granted`
  - user details are shown
- If face is detected but does not match:
  - `Face Detected - Access Denied (Unknown User)`
  - no user details are shown
- If no face is detected:
  - `No Face Detected`

## Installation

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the application

```powershell
python app.py
```

Then open:

- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## First Run

1. Open the app
2. Create the first admin account
3. Log in to the admin portal
4. Add users and capture `10-20` samples each
5. Train the model
6. Open the authentication page and test recognition

## Manual Training

You can also train the model from the terminal:

```powershell
python train.py
```

## Notes

- The current build stores processed face vectors, not raw face images, for the main enrollment workflow
- MediaPipe facial landmarks are optional; if MediaPipe is not installed, the system falls back to OpenCV eye detection
- Optional anti-spoofing is not enabled yet in this version

## Recommended Next Upgrade

- Add blink detection or head-movement anti-spoofing
- Add downloadable admin reports
- Move from SQLite to MySQL if multi-user deployment is needed
