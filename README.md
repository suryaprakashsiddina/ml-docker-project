# 🚀 ML Model Prediction App — Flask + Docker

A lightweight **machine learning model deployment project** that exposes a trained ML model through a **Flask web application and REST API**, with the entire application containerized using **Docker**.

The application loads a pre-trained machine learning model, accepts an input value from either a browser-based interface or an API request, performs prediction, and returns the result.

---

## 📌 Overview

This project demonstrates how a trained machine learning model can be converted into a usable application and deployed in a containerized environment.

The workflow is:

```text
User Input
    │
    ├──────────────► Web Interface
    │
    └──────────────► REST API
                         │
                         ▼
                  Flask Application
                         │
                         ▼
                    model.pkl
                         │
                         ▼
                 ML Model Prediction
                         │
                         ▼
                  Prediction Result
```

---

## ✨ Features

* 🧠 Loads a pre-trained machine learning model using Pickle.
* 🌐 Provides a simple browser-based prediction interface.
* 🔌 Provides a REST API for programmatic predictions.
* 🐍 Built using Python and Flask.
* 🐳 Containerized using Docker.
* 📦 Uses `requirements.txt` for dependency management.
* 🔄 Supports both form-based and JSON-based prediction requests.
* 🚀 Runs on port `5000`.

---

## 🛠️ Tech Stack

| Technology   | Purpose                                |
| ------------ | -------------------------------------- |
| Python       | Application development                |
| Flask        | Web framework and REST API             |
| NumPy        | Input preparation for model prediction |
| Pickle       | Loading the trained ML model           |
| Docker       | Application containerization           |
| Scikit-learn | Machine learning model support         |
| Git/GitHub   | Version control                        |

---

## 📂 Project Structure

```text
ml-docker-project/
│
├── Dockerfile
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

### File Description

#### `app.py`

The main Flask application.

It:

* Loads the trained model.
* Creates the web interface.
* Handles prediction requests.
* Supports browser form submissions.
* Supports JSON API requests.
* Returns prediction results.

#### `model.pkl`

Serialized pre-trained machine learning model used to generate predictions.

#### `requirements.txt`

Contains the Python dependencies required to run the application.

#### `Dockerfile`

Contains the instructions required to build the Docker image for the application.

---

## 🏗️ Application Architecture

```text
                    ┌──────────────────┐
                    │      Client      │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │                  │
                    ▼                  ▼
             Web Form Request     JSON API Request
                    │                  │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │ Flask App       │
                    │    app.py       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   model.pkl     │
                    │ Trained ML Model│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Prediction    │
                    └─────────────────┘
```

---

## 🌐 Web Interface

The application provides a simple web interface where users can enter an input value and receive a prediction.

```text
┌─────────────────────────────────────┐
│       ML Model Prediction App 🚀    │
│                                     │
│  This app predicts output based     │
│  on the trained ML model.           │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ Enter value                   │  │
│  └───────────────────────────────┘  │
│                                     │
│             [ Predict ]             │
│                                     │
│        Prediction: <result>         │
└─────────────────────────────────────┘
```

The home page is available at:

```text
http://localhost:5000/
```

---

## 🔌 REST API

The application also exposes a prediction endpoint.

### Endpoint

```text
POST /predict
```

### JSON Request

Send a JSON request containing an input value:

```json
{
  "input": 10
}
```

### Example using cURL

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d "{\"input\": 10}"
```

### Example Response

```json
{
  "prediction": 1
}
```

> The actual prediction depends on the trained model stored in `model.pkl`.

---

## 🔄 Prediction Process

The prediction logic follows these steps:

```text
Input Value
     ↓
Convert to Float
     ↓
Create NumPy Array
     ↓
Pass Input to ML Model
     ↓
model.predict()
     ↓
Extract Prediction
     ↓
Return Result
```

Internally, the input is prepared in the format:

```python
np.array([[input_value]])
```

This allows the input to be passed to the trained machine learning model for inference.

---

# 🐳 Docker Deployment

One of the main goals of this project is to demonstrate how an ML application can be **containerized using Docker**.

Instead of installing all dependencies directly on the host machine, the application and its required environment can be packaged into a Docker image.

```text
Flask Application
       +
ML Model
       +
Python Dependencies
       ↓
   Docker Image
       ↓
 Docker Container
       ↓
Running ML API
```

---

## ⚙️ Run Locally Without Docker

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/ml-docker-project.git
```

### 2. Navigate into the project

```bash
cd ml-docker-project
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Flask application

```bash
python app.py
```

The application will start on:

```text
http://localhost:5000
```

Open the URL in your browser to access the prediction interface.

---

# 🐳 Run Using Docker

Make sure Docker is installed and running on your system.

### 1. Build the Docker image

From the project directory:

```bash
docker build -t ml-model-prediction .
```

### 2. Run the container

```bash
docker run -p 5000:5000 ml-model-prediction
```

The application can then be accessed at:

```text
http://localhost:5000
```

---

## 🧪 Testing the API

Once the Docker container is running, the API can be tested using cURL, Postman, or another API client.

### Example

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d "{\"input\": 5}"
```

Expected response format:

```json
{
  "prediction": "<model output>"
}
```

---

## 🔐 Error Handling

The prediction endpoint includes exception handling to prevent unexpected input or model errors from crashing the application.

The application attempts to handle both:

```text
Browser Form Input
        +
JSON API Input
```

through the same `/predict` endpoint.

---

## 📊 Deployment Workflow

The complete deployment workflow is:

```text
Train ML Model
      ↓
Save Model as model.pkl
      ↓
Create Flask Application
      ↓
Create Prediction API
      ↓
Create Dockerfile
      ↓
Build Docker Image
      ↓
Run Docker Container
      ↓
Access ML Application / API
```

---

## 💡 What This Project Demonstrates

This project demonstrates practical understanding of the **ML model deployment lifecycle**.

### Machine Learning

* Loading a trained ML model
* Preparing input data
* Performing model inference
* Returning predictions

### Backend Development

* Flask application development
* HTTP routing
* POST requests
* JSON request/response handling

### API Development

* REST-style prediction endpoint
* JSON-based model inference
* API testing using tools such as cURL/Postman

### MLOps / Deployment

* Dependency management
* Model packaging
* Docker containerization
* Running ML services inside containers

---

## 🔮 Future Improvements

Potential improvements include:

* Add input validation with meaningful error messages.
* Add multiple model input features.
* Add prediction confidence/probability.
* Add Swagger/OpenAPI documentation.
* Add automated unit and API tests.
* Add logging and monitoring.
* Add Docker Compose configuration.
* Add CI/CD using GitHub Actions.
* Deploy the container to AWS, Azure, or GCP.
* Add a production WSGI server such as Gunicorn.
* Add model versioning.
* Add MLflow for experiment and model tracking.
* Add authentication for production API usage.

---

## 📌 Project Highlights

* Built a **Flask-based machine learning prediction application**.
* Integrated a pre-trained ML model using `model.pkl`.
* Created both a **web-based prediction interface and REST API**.
* Implemented JSON-based model inference.
* Containerized the ML application using **Docker**.
* Demonstrated the transition from a trained ML model to a deployable application.
* Designed the project as a foundation for scalable ML model serving.

---

## 👨‍💻 Author

**Surya Prakash Siddina**

M.Tech — Artificial Intelligence & Data Science
B.Tech — Computer Science & Engineering

### Areas of Interest

* Artificial Intelligence
* Machine Learning
* MLOps
* Backend Development
* Generative AI
* Data Science
* Software Development

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
