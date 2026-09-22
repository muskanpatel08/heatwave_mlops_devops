# Heatwave Prediction — ML, MLflow, MLOps and DevOps

Pipeline:
Dataset -> Machine Learning -> MLflow -> Flask API -> Docker -> GitHub Actions

Target:
Today's weather is used to predict tomorrow's heatwave flag.

API:
GET /health
POST /predict

Local:
pip install -r requirements.txt
python app.py

Docker:
docker build -t heatwave-api:1.0 .
docker run -p 8000:8000 heatwave-api:1.0
