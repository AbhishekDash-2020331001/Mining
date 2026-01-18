import os
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Optional

class ModelManager:
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.model_info_path = self.models_dir / "model_info.json"
        self.model_params_path = self.models_dir / "model_params.json"
        self.training_csv_path = self.models_dir / "training_data.csv"
    
    def model_exists(self) -> bool:
        """Check if trained model files exist"""
        return (
            self.model_info_path.exists() and
            (self.models_dir / "trained_models.joblib").exists()
        )
    
    def save_model_info(self, info: dict):
        """Save model metadata"""
        with open(self.model_info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def save_model_params(self, params: dict):
        """Save feature parameters"""
        with open(self.model_params_path, 'w') as f:
            json.dump(params, f, indent=2)
    
    def save_models(self, models_dict: dict):
        """Save trained models"""
        joblib.dump(models_dict, self.models_dir / "trained_models.joblib")
    
    def get_model_info(self) -> dict:
        """Get model metadata"""
        if not self.model_info_path.exists():
            return {}
        with open(self.model_info_path, 'r') as f:
            return json.load(f)
    
    def get_model_params(self) -> dict:
        """Get feature parameters"""
        if not self.model_params_path.exists():
            return {}
        with open(self.model_params_path, 'r') as f:
            return json.load(f)
    
    def load_models(self) -> dict:
        """Load trained models"""
        model_path = self.models_dir / "trained_models.joblib"
        if not model_path.exists():
            return {}
        return joblib.load(model_path)
    
    def save_training_csv(self, csv_content: bytes, filename: Optional[str] = None):
        """Save the training CSV file"""
        self.training_csv_path.write_bytes(csv_content)
        # Store metadata
        info = self.get_model_info()
        if "csv_info" not in info:
            info["csv_info"] = {}
        info["csv_info"]["filename"] = filename or "training_data.csv"
        info["csv_info"]["upload_date"] = datetime.now().isoformat()
        info["csv_info"]["size_bytes"] = len(csv_content)
        info["csv_info"]["rows"] = None  # Can be calculated if needed
        self.save_model_info(info)
    
    def get_training_csv(self) -> Optional[bytes]:
        """Get the training CSV file content"""
        if self.training_csv_path.exists():
            return self.training_csv_path.read_bytes()
        return None
    
    def get_csv_info(self) -> Optional[dict]:
        """Get CSV metadata"""
        info = self.get_model_info()
        return info.get("csv_info")
    
    def clear_models(self):
        """Clear all saved models"""
        for file in self.models_dir.glob("*"):
            if file.is_file():
                file.unlink()

