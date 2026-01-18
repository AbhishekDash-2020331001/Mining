import numpy as np
from typing import Dict, Any
from app.services.features import FeatureEngineering
from app.services.model_manager import ModelManager

class PredictionService:
    """
    Service to make PPV predictions using trained models.
    """
    
    def __init__(self):
        self.feature_eng = FeatureEngineering()
        self.model_manager = ModelManager()
        self.models = None
        self.feature_params = None
        self.model_info = None
        self.train_mae = None
    
    def _load_models_if_needed(self):
        """Lazy loading of models"""
        if self.models is None:
            self.models = self.model_manager.load_models()
            self.feature_params = self.model_manager.get_model_params()
            info = self.model_manager.get_model_info()
            if info:
                self.model_info = info.get("model_info", {})
                # Extract train_mae from metrics if available
                metrics = info.get("metrics", {})
                if "test" in metrics:
                    self.train_mae = metrics["test"].get("train_mae")
    
    def predict(self, R: float, W: float) -> Dict[str, Any]:
        """
        Predict PPV for given R and W values.
        Returns predicted PPV, uncertainty range, and computed features.
        """
        self._load_models_if_needed()
        
        if not self.models or len(self.models) == 0:
            raise ValueError("No trained models found")
        
        # Compute all features using saved parameters
        try:
            features_dict = self.feature_eng.compute_features_from_params(W, R, self.feature_params)
        except Exception as e:
            raise ValueError(
                f"Error computing features: {str(e)}. "
                f"Feature params available: {list(self.feature_params.keys()) if self.feature_params else 'None'}"
            ) from e
        
        # Get predictions from all models
        predictions = []
        
        for model_key, model_data in self.models.items():
            model = model_data["model"]
            stored_feature_list = model_data["features"]
            
            # Try to get actual feature names from the model's StandardScaler
            # This is more reliable than the stored feature list
            actual_feature_list = stored_feature_list
            try:
                if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                    scaler = model.named_steps['scaler']
                    if hasattr(scaler, 'feature_names_in_'):
                        actual_feature_list = list(scaler.feature_names_in_)
            except Exception:
                # If we can't extract from scaler, use stored list
                pass
            
            # Use actual feature list from model if available, otherwise use stored
            feature_list = actual_feature_list if actual_feature_list != stored_feature_list else stored_feature_list
            
            # Check if all required features are available
            missing_features = [feat for feat in feature_list if feat not in features_dict]
            if missing_features:
                raise ValueError(
                    f"Missing features: {missing_features}. "
                    f"Available features: {list(features_dict.keys())}. "
                    f"Required features: {feature_list}. "
                    f"Model key: {model_key}. "
                    f"This might be due to a model format mismatch. Please retrain the model."
                )
            
            # Create feature vector for this model as DataFrame with column names
            # IMPORTANT: Use the exact same feature order as during training
            try:
                import pandas as pd
                # Create DataFrame with features in the exact order they were used during training
                # Use feature_list order to ensure consistency with training
                feature_values = [features_dict[feat] for feat in feature_list]
                feature_df = pd.DataFrame([feature_values], columns=feature_list)
                
                # Get prediction
                pred = model.predict(feature_df)[0]
                predictions.append(pred)
            except Exception as e:
                # Enhanced error message to help debug feature mismatch
                error_msg = str(e)
                if "feature names" in error_msg.lower() or "fit time" in error_msg.lower():
                    raise ValueError(
                        f"Feature mismatch error with model {model_key}: {error_msg}. "
                        f"Stored feature list: {stored_feature_list}. "
                        f"Actual model features (from scaler): {actual_feature_list}. "
                        f"Available features: {list(features_dict.keys())}. "
                        f"This suggests the model was saved with incorrect feature information. "
                        f"Please retrain the model."
                    ) from e
                else:
                    raise ValueError(
                        f"Error predicting with model {model_key}: {error_msg}. "
                        f"Feature list: {feature_list}, Available features: {list(features_dict.keys())}"
                    ) from e
        
        # Ensemble prediction (average)
        predicted_ppv = float(np.mean(predictions))
        
        # Calculate uncertainty range if train_mae is available
        min_range = None
        max_range = None
        
        if self.train_mae is not None:
            alpha = 1.8  # safety factor (same as training)
            min_range = float(predicted_ppv - alpha * self.train_mae)
            max_range = float(predicted_ppv + alpha * self.train_mae)
        
        return {
            "predicted_ppv": predicted_ppv,
            "min_range": min_range,
            "max_range": max_range,
            "features": features_dict
        }

