from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
import pandas as pd
from app.api.schemas import PredictionRequest, PredictionResponse, TrainingResponse, ModelStatusResponse
from app.services.training import TrainingService
from app.services.prediction import PredictionService
from app.services.model_manager import ModelManager
import io
import zipfile
import tempfile
from pathlib import Path

router = APIRouter()
training_service = TrainingService()
prediction_service = PredictionService()
model_manager = ModelManager()

@router.post("/train", response_model=TrainingResponse)
async def train_model(file: UploadFile = File(...)):
    """
    Upload CSV file and train the mining prediction model.
    CSV should contain columns: R, W, PPV
    """
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate CSV format
        required_columns = ['R', 'W', 'PPV']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain columns: {', '.join(required_columns)}"
            )
        
        # Train model (save CSV content and filename)
        csv_filename = file.filename or "training_data.csv"
        result = training_service.train_model(df, csv_content=contents, csv_filename=csv_filename)
        
        # Create message with cleaning info
        cleaning_info = result.get("cleaning_info", {})
        original_rows = cleaning_info.get("original_rows", 0)
        nan_dropped = cleaning_info.get("nan_dropped", 0)
        zero_dropped = cleaning_info.get("zero_dropped", 0)
        final_rows = cleaning_info.get("final_rows", 0)
        
        message = "Model trained successfully"
        if nan_dropped > 0 or zero_dropped > 0:
            message += f". Cleaned data: {original_rows} → {final_rows} rows"
            if nan_dropped > 0:
                message += f" (removed {nan_dropped} with NaN"
            if zero_dropped > 0:
                if nan_dropped > 0:
                    message += f", {zero_dropped} with zero"
                else:
                    message += f" (removed {zero_dropped} with zero"
            if nan_dropped > 0 or zero_dropped > 0:
                message += ")"
        
        return TrainingResponse(
            status="success",
            message=message,
            metrics=result.get("metrics"),
            model_info=result.get("model_info")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=PredictionResponse)
async def predict_ppv(request: PredictionRequest):
    """
    Predict PPV value given R and W values.
    """
    try:
        # Check if model exists
        if not model_manager.model_exists():
            raise HTTPException(
                status_code=404,
                detail="No trained model found. Please train a model first."
            )
        
        # Validate inputs
        if request.R <= 0 or request.W <= 0:
            raise HTTPException(
                status_code=400,
                detail="R and W must be positive values"
            )
        
        # Get prediction
        result = prediction_service.predict(request.R, request.W)
        
        return PredictionResponse(
            predicted_ppv=result["predicted_ppv"],
            min_range=result.get("min_range"),
            max_range=result.get("max_range"),
            features=result.get("features")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Check if a trained model exists.
    """
    exists = model_manager.model_exists()
    info = None
    if exists:
        full_info = model_manager.get_model_info()
        # Extract just the model_info part from the saved structure
        info = full_info.get("model_info") if full_info else None
    
    return ModelStatusResponse(
        model_exists=exists,
        model_info=info
    )

@router.get("/training-csv")
async def get_training_csv():
    """
    Download the CSV file used for training.
    """
    try:
        csv_content = model_manager.get_training_csv()
        
        if csv_content is None:
            raise HTTPException(
                status_code=404,
                detail="No training CSV file found"
            )
        
        csv_info = model_manager.get_csv_info()
        filename = csv_info.get("filename", "training_data.csv") if csv_info else "training_data.csv"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-csv/info")
async def get_training_csv_info():
    """
    Get metadata about the training CSV file.
    """
    try:
        csv_info = model_manager.get_csv_info()
        
        if csv_info is None:
            raise HTTPException(
                status_code=404,
                detail="No training CSV file found"
            )
        
        return csv_info
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "service": "mining-prediction-api",
        "model_exists": model_manager.model_exists()
    }

@router.get("/models/download")
async def download_models():
    """
    Download all model files as a ZIP bundle for local storage.
    Includes: trained_models.joblib, model_info.json, model_params.json, training_data.csv
    """
    try:
        if not model_manager.model_exists():
            raise HTTPException(
                status_code=404,
                detail="No trained model found. Please train a model first."
            )
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        zip_path = temp_zip.name
        temp_zip.close()
        
        # Create ZIP archive with all model files
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            models_dir = model_manager.models_dir
            
            # Add model files if they exist
            files_to_add = [
                ("trained_models.joblib", "trained_models.joblib"),
                ("model_info.json", "model_info.json"),
                ("model_params.json", "model_params.json"),
                ("training_data.csv", "training_data.csv")
            ]
            
            for filename, zip_name in files_to_add:
                file_path = models_dir / filename
                if file_path.exists():
                    zipf.write(file_path, zip_name)
        
        # Read ZIP file content
        with open(zip_path, 'rb') as f:
            zip_content = f.read()
        
        # Clean up temp file
        Path(zip_path).unlink()
        
        return Response(
            content=zip_content,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=mining_model_bundle.zip"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
