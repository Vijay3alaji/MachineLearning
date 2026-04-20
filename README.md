# 🎓 Student Exam Performance Predictor

> An end-to-end Machine Learning web application that predicts a student's **Mathematics score** based on demographic, socioeconomic, and academic input features — built, containerized, and deployed to production.

**🔴 Live Demo → [machine-learning.vercel.app](https://machine-learning-tawny.vercel.app/)**

---

## 📸 Screenshots

| Landing Page | Prediction Form | Result |
|---|---|---|
| ![Home](<img width="745" height="948" alt="image" src="https://github.com/user-attachments/assets/cf66fd5e-a1ca-42c7-8a52-574d375ca154" />
) | ![Form](<img width="1148" height="871" alt="image" src="https://github.com/user-attachments/assets/74c4ab28-90d6-4837-8ac7-83ae3e3db5da" />
) | ![Result]() |


---

## 📌 Table of Contents

- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [ML Pipeline](#-ml-pipeline)
- [Features](#-features)
- [Getting Started](#-getting-started)
- [Docker](#-docker)
- [Deployment](#-deployment)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Author](#-author)

---

## 🧠 Overview

This project demonstrates a **complete, production-grade ML lifecycle** — from raw data ingestion and model training to a Flask web application deployed live on Vercel and containerized with Docker.

The model takes 7 student attributes as input and predicts the expected mathematics exam score (out of 100) using a regression algorithm trained on the [Students Performance Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

This is not a notebook project — it is structured as a **real-world software engineering project** with modular code, custom exception handling, logging, a prediction pipeline, and CI/CD via GitHub → Vercel.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Web Framework** | Flask |
| **ML Library** | Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Model Serialization** | Dill |
| **Frontend** | HTML5, CSS3, Jinja2 |
| **Containerization** | Docker |
| **Container Registry** | Docker Hub |
| **Deployment** | Vercel (Serverless) |
| **Version Control** | Git + GitHub |

---

## 🏗 Project Architecture

```
User (Browser)
      │
      ▼
  Vercel (Serverless)
      │
      ▼
  Flask App (app.py)
      │
      ├──► / ──────────────────► index.html (Landing Page)
      │
      └──► /predict
              │
              ├── GET  ────────► home.html (Input Form)
              │
              └── POST
                    │
                    ▼
              CustomData Object
                    │
                    ▼
              PredictPipeline
                    │
                    ├── Load preprocessor.pkl  (StandardScaler + OrdinalEncoder)
                    ├── Transform input features
                    └── Load model.pkl → Predict score
                    │
                    ▼
              home.html (Result Display)
```

---

## 🔬 ML Pipeline

The ML pipeline is modular and follows production best practices:

**1. Data Ingestion** — Reads raw CSV data and splits it into train/test sets, saving them to the `artifacts/` directory.

**2. Data Transformation** — Builds a `ColumnTransformer` pipeline with `StandardScaler` for numerical features and `OrdinalEncoder` for categorical features. The fitted preprocessor is saved as `preprocessor.pkl`.

**3. Model Training** — Trains and evaluates multiple regression models including Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost, CatBoost, and AdaBoost. The best performing model is saved as `model.pkl`.

**4. Prediction Pipeline** — `PredictPipeline` loads both artifacts at inference time and transforms + predicts on new user input.

```
Raw Input (7 features)
       │
       ▼
preprocessor.pkl  ──►  Encoded + Scaled Features
       │
       ▼
   model.pkl  ──────►  Predicted Math Score (float)
```

---

## ✨ Features

- **End-to-end ML pipeline** — ingestion → transformation → training → serving
- **Modular codebase** — clean separation of concerns across `src/` modules
- **Custom exception handling** — descriptive errors with file name and line number
- **Logging** — timestamped log files generated at runtime
- **Interactive UI** — slider + number input for scores, instant result with performance badge
- **Dockerized** — runs identically in any environment with a single command
- **Live deployment** — publicly accessible on Vercel with zero manual steps on push

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Vijay3alaji/MachineLearning.git
cd MachineLearning
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Open your browser and go to `http://localhost:5000`

---

## 🐳 Docker

The application is fully containerized. Pull and run the pre-built image from Docker Hub:

```bash
# Pull the image
docker pull vijay3alaji/student-exam-predictor:latest

# Run the container
docker run -p 5000:5000 vijay3alaji/student-exam-predictor:latest
```

Or build it yourself locally:

```bash
docker build -t student-exam-predictor .
docker run -p 5000:5000 student-exam-predictor
```

**Docker Hub →** `https://hub.docker.com/r/vijay3alaji/student-exam-predictor`

---

## ☁️ Deployment

The app is deployed on **Vercel** using serverless functions via `vercel.json`. Every push to the `main` branch triggers an automatic redeploy.

```json
{
  "version": 2,
  "builds": [{ "src": "app.py", "use": "@vercel/python" }],
  "routes": [{ "src": "/(.*)", "dest": "app.py" }]
}
```

**Live URL →** `https://machine-learning.vercel.app`

---

## 📁 Project Structure

```
MachineLearning/
│
├── app.py                          # Flask application entry point
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker image definition
├── vercel.json                     # Vercel deployment config
│
├── artifacts/                      # Saved ML artifacts (generated)
│   ├── model.pkl                   # Trained best model
│   ├── preprocessor.pkl            # Fitted data transformer
│   ├── train.csv
│   └── test.csv
│
├── src/                            # Core source modules
│   ├── __init__.py
│   ├── exception.py                # Custom exception class
│   ├── logging_config.py           # Logging setup
│   ├── utils.py                    # Shared utility functions
│   │
│   ├── components/                 # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   └── pipeline/
│       └── predict_pipeline.py     # Inference pipeline
│
├── templates/                      # Jinja2 HTML templates
│   ├── index.html                  # Landing page
│   └── home.html                   # Prediction form + result
│
└── notebook/                       # EDA and experimentation
    └── EDA.ipynb
```

---

## 📊 Results

| Model | R² Score |
|---|---|
| Linear Regression | 0.879 |
| Ridge Regression | 0.881 |
| **Gradient Boosting** | **0.888** |
| Random Forest | 0.855 |
| XGBoost | 0.868 |
| CatBoost | 0.889 |
| AdaBoost | 0.850 |

> The best model was selected automatically based on R² score and saved to `artifacts/model.pkl`.

_Update the table above with your actual model evaluation results from `model_trainer.py`_

---

## 💡 Input Features

| Feature | Type | Example Values |
|---|---|---|
| Gender | Categorical | male, female |
| Race/Ethnicity | Categorical | group A, B, C, D, E |
| Parental Education | Categorical | bachelor's degree, high school, etc. |
| Lunch Type | Categorical | standard, free/reduced |
| Test Prep Course | Categorical | completed, none |
| Reading Score | Numerical | 0 – 100 |
| Writing Score | Numerical | 0 – 100 |

**Output:** Predicted Mathematics Score (0 – 100, continuous)

---

## 👤 Author

**Vijay Alaji**

[![GitHub](https://img.shields.io/badge/GitHub-Vijay3alaji-181717?style=for-the-badge&logo=github)](https://github.com/Vijay3alaji)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/vijay-balajim/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built end-to-end by Vijay Alaji — from raw data to live production deployment</sub>
</div>
