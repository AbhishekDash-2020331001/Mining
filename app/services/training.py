import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from itertools import combinations
from typing import Dict, List, Tuple, Any, Optional
from app.services.features import FeatureEngineering
from app.services.model_manager import ModelManager

class TrainingService:
    """
    Service to train the mining prediction model.
    Replicates the training logic from mining2.ipynb
    """
    
    def __init__(self):
        self.feature_eng = FeatureEngineering()
        self.model_manager = ModelManager()
        
        # Define models (same as notebook)
        self.models = {
            "LinearRegression_OLS": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ]),
            "Ridge_L2": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42))
            ]),
            "Lasso_L1": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.01, max_iter=10000, random_state=42))
            ]),
            "ElasticNet_L1L2": Pipeline([
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000, random_state=42))
            ]),
        }
    
    def train_model(self, df: pd.DataFrame, csv_content: Optional[bytes] = None, csv_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the mining prediction model.
        Returns metrics and model info.
        """
        # Remove rows with zero values
        df_clean = df[(df != 0).all(axis=1)].copy()
        
        # Split data: 75% train, 12.5% val, 12.5% test
        train_df, temp_df = train_test_split(
            df_clean, test_size=0.25, random_state=44, shuffle=True
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=44, shuffle=True
        )
        
        # Generate all features
        train_features, feature_params = self.feature_eng.generate_all_features(train_df)
        val_features, _ = self.feature_eng.generate_all_features(val_df)
        test_features, _ = self.feature_eng.generate_all_features(test_df)
        
        # Add features to dataframes
        for feature_name, feature_values in train_features.items():
            train_df[feature_name] = feature_values
        for feature_name, feature_values in val_features.items():
            val_df[feature_name] = feature_values
        for feature_name, feature_values in test_features.items():
            test_df[feature_name] = feature_values
        
        # Get all feature names (excluding PPV)
        all_features = list(train_df.drop(columns=['PPV']).columns)
        
        # Generate all 4-feature combinations
        feature_sets = list(combinations(all_features, 4))
        
        # Evaluate models on train set
        train_results = []
        for fset in feature_sets:
            X_train_sub = train_df[list(fset)]
            y_train = train_df["PPV"]
            
            row = {"Features": fset}
            for name, model in self.models.items():
                model.fit(X_train_sub, y_train)
                y_pred = model.predict(X_train_sub)
                r2 = r2_score(y_train, y_pred)
                row[name] = r2
            train_results.append(row)
        
        train_results_df = pd.DataFrame(train_results)
        
        # Evaluate models on validation set
        val_results = []
        for fset in feature_sets:
            X_train_sub = train_df[list(fset)]
            X_val_sub = val_df[list(fset)]
            y_train = train_df["PPV"]
            y_val = val_df["PPV"]
            
            row = {"Features": fset}
            for name, model in self.models.items():
                model.fit(X_train_sub, y_train)
                y_pred = model.predict(X_val_sub)
                r2 = r2_score(y_val, y_pred)
                row[name] = r2
            val_results.append(row)
        
        val_results_df = pd.DataFrame(val_results)
        
        # Select best models (merge train and val results)
        model_cols = train_results_df.columns.drop("Features")
        train_long = train_results_df.melt(
            id_vars="Features", value_vars=model_cols, 
            var_name="model_name", value_name="trainR2"
        )
        val_long = val_results_df.melt(
            id_vars="Features", value_vars=model_cols,
            var_name="model_name", value_name="valR2"
        )
        
        merged = pd.merge(train_long, val_long, on=["Features", "model_name"])
        merged["R2gap"] = merged["valR2"] - merged["trainR2"]
        
        # Filter models with small gap (R2gap >= -1.0)
        best_models = merged[merged["R2gap"] >= -1.0].copy()
        best_models = best_models.sort_values(by="trainR2", ascending=False)
        
        # Select top 5 models
        best_models5 = best_models.head(5)
        
        if len(best_models5) == 0:
            # Fallback: use top models regardless of gap
            best_models5 = merged.sort_values(by="trainR2", ascending=False).head(5)
        
        # Train final models and get ensemble predictions
        trained_models_dict = {}
        train_preds = []
        val_preds = []
        test_preds = []
        
        for _, row in best_models5.iterrows():
            feats = list(row["Features"])
            model_name = row["model_name"]
            
            # Create a fresh model instance
            model = self.models[model_name]
            model.fit(train_df[feats], train_df["PPV"])
            
            # Store the model with its feature set
            model_key = f"{model_name}_{'_'.join(sorted(feats))}"
            trained_models_dict[model_key] = {
                "model": model,
                "features": feats,
                "model_name": model_name
            }
            
            # Get predictions
            train_preds.append(model.predict(train_df[feats]))
            val_preds.append(model.predict(val_df[feats]))
            test_preds.append(model.predict(test_df[feats]))
        
        # Ensemble predictions (average)
        train_prediction = np.mean(train_preds, axis=0)
        val_prediction = np.mean(val_preds, axis=0)
        test_prediction = np.mean(test_preds, axis=0)
        
        # Calculate metrics
        train_r2 = r2_score(train_df["PPV"], train_prediction)
        val_r2 = r2_score(val_df["PPV"], val_prediction)
        test_r2 = r2_score(test_df["PPV"], test_prediction)
        
        train_rmse = np.sqrt(mean_squared_error(train_df["PPV"], train_prediction))
        val_rmse = np.sqrt(mean_squared_error(val_df["PPV"], val_prediction))
        test_rmse = np.sqrt(mean_squared_error(test_df["PPV"], test_prediction))
        
        train_mae = mean_absolute_error(train_df["PPV"], train_prediction)
        val_mae = mean_absolute_error(val_df["PPV"], val_prediction)
        test_mae = mean_absolute_error(test_df["PPV"], test_prediction)
        
        train_mae_pct = (train_mae / train_df["PPV"].mean()) * 100
        test_mae_pct = (test_mae / test_df["PPV"].mean()) * 100
        
        # Calculate additional metrics
        metrics = {
            "train": {
                "r2": float(train_r2),
                "rmse": float(train_rmse),
                "mae": float(train_mae),
                "mae_pct": float(train_mae_pct)
            },
            "validation": {
                "r2": float(val_r2),
                "rmse": float(val_rmse),
                "mae": float(val_mae)
            },
            "test": {
                "r2": float(test_r2),
                "rmse": float(test_rmse),
                "mae": float(test_mae),
                "mae_pct": float(test_mae_pct)
            }
        }
        
        # Calculate uncertainty bands (for test set)
        errors_train = np.abs(train_df["PPV"].values - train_prediction)
        train_mae_value = np.mean(errors_train)
        alpha = 1.8  # safety factor
        test_min = test_prediction - alpha * train_mae_value
        test_max = test_prediction + alpha * train_mae_value
        
        # Coverage ratio
        coverage = np.mean(
            (test_df["PPV"].values >= test_min) & (test_df["PPV"].values <= test_max)
        )
        
        metrics["test"]["coverage_ratio"] = float(coverage)
        metrics["test"]["train_mae"] = float(train_mae_value)
        
        # Model info
        model_info = {
            "num_models": len(best_models5),
            "selected_models": [
                {
                    "model_name": row["model_name"],
                    "features": list(row["Features"]),
                    "train_r2": float(row["trainR2"]),
                    "val_r2": float(row["valR2"]),
                    "r2_gap": float(row["R2gap"])
                }
                for _, row in best_models5.iterrows()
            ],
            "total_features": len(all_features),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df)
        }
        
        # Save models and parameters
        self.model_manager.save_models(trained_models_dict)
        self.model_manager.save_model_params(feature_params)
        
        # Prepare model info dict
        model_info_dict = {
            "model_info": model_info,
            "metrics": metrics,
            "best_models": [
                {
                    "model_name": row["model_name"],
                    "features": list(row["Features"])
                }
                for _, row in best_models5.iterrows()
            ]
        }
        
        # Save CSV file if provided (this adds csv_info to model_info)
        if csv_content is not None:
            self.model_manager.save_training_csv(csv_content, csv_filename)
            # Get the csv_info that was just saved and merge it into model_info_dict
            saved_info = self.model_manager.get_model_info()
            if "csv_info" in saved_info:
                model_info_dict["csv_info"] = saved_info["csv_info"]
        
        # Save model info (with csv_info if it was added)
        self.model_manager.save_model_info(model_info_dict)
        
        return {
            "metrics": metrics,
            "model_info": model_info
        }

