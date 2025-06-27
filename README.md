# Multi-Modal Person Identification System

This project implements a multi-modal system for identifying individuals using both image and audio inputs. It features a FastAPI backend for inference and a React-based web frontend for interaction.

## Features

* **Multimodal Input:** Processes both facial images and voice recordings.
* **Deep Learning Models:** Utilizes PyTorch models (CNN for images, RNN for audio, Attention for fusion) for robust identification.
* **RESTful API:** Provides an easy-to-use `/predict` endpoint via FastAPI.
* **Interactive Web Frontend:** A simple browser interface for capturing and submitting live camera and microphone data.

## Technologies Used

* **Backend:**
    * Python 3.x
    * FastAPI
    * PyTorch
    * Uvicorn
    * Pillow, torchaudio, numpy
* **Frontend:**
    * React.js
    * Vite
    * JavaScript (ES6+)
    * HTML5, CSS3

## Project Structure
```bash
.
├── backend
│  ├── app
│  │  ├── attention_fusion_model.pth
│  │  └── main.py
│  ├── requirements.txt
│  └── venv
├── create_dir.sh
├── dataset
├── frontend
│  ├── eslint.config.js
│  ├── index.html
│  ├── node_modules
│  ├── package-lock.json
│  ├── package.json
│  ├── public
│  │  └── vite.svg
│  ├── README.md
│  ├── src
│  │  ├── App.css
│  │  ├── App.jsx
│  │  ├── assets
│  │  ├── index.css
│  │  └── main.jsx
│  └── vite.config.js
├── LICENSE
└── notebooks
   └── botchSec.ipynb

```

## Set up Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:tchaikovsky1984/botchSecurity.git
cd botchSecurity
```

### 2. Prepare the Dataset

Run the following bash script with N = number of classes.
```bash
bash create_dir.sh N
```
Now, you may mv in the data.

### 3. Train and Place the model

Edit the ```notebooks/botchSec.ipynb``` file to save the ```attention_fusion_model.pth``` file to ```backend/app/```.
Train the model on your data.
Write the model state to the aforementioed location.

### 4. Backend Set Up

```bash
cd backend
pip install -r requirements.txt
```

### 5. Frontend Set Up

```bash
cd frontend
npm install
```

## Running

### 1. Run the Backend

```bash
cd backend/app
uvicorn main:app --reload
```

### 2. Run the Frontend

```bash
cd frontend
npm run dev
```

## Access Points

### 1. Frontend

http://localhost:5173/

### 2. FastAPI Docs

http://localhost:8000/docs

## Testing the Backend

### 1. Run the Backend

```bash
cd backend/app/
uvicorn main:app --reload
```

### 2. Curl 

```bash
curl -X POST "[http://127.0.0.1:8000/predict/](http://127.0.0.1:8000/predict/)" -H "accept: application/json" -F "image_file=@/path/to/your/dataset/person_name/images/image.jpg;type=image/jpeg" -F "audio_file=@/path/to/your/dataset/person_name/audio/audio.mp3;type=audio/mpeg"
```

## Contributions

Built as a speedrun for a class. Contributions are welcome. For big updates, please open an issue.
