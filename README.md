# Mining Prediction API Backend

FastAPI backend for mining PPV prediction model training and inference.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Access API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### POST /api/train
Upload a CSV file with R, W, PPV columns to train the model.

**Request**: multipart/form-data with file upload

**Response**: Training metrics and model information

### POST /api/predict
Predict PPV value given R and W.

**Request Body**:
```json
{
  "R": 10.5,
  "W": 2.3
}
```

**Response**:
```json
{
  "predicted_ppv": 0.045,
  "min_range": 0.040,
  "max_range": 0.050,
  "features": {...}
}
```

### GET /api/model/status
Check if a trained model exists.

**Response**:
```json
{
  "model_exists": true,
  "model_info": {...}
}
```

## Model Storage

Trained models are saved in the `models/` directory:
- `trained_models.joblib`: Saved scikit-learn models
- `model_params.json`: Feature engineering parameters
- `model_info.json`: Model metadata and metrics

