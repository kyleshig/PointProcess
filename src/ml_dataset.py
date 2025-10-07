
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

class TennisMLPipeline:
    """Tennis match predictor baseline using logistic regression"""

    def __init__(self, training_df):
        """Initialize with pre-built training DataFrame from TennisFeatureEngineer"""

        self.training_df = training_df
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()

        # Extract only numeric features (exclude target, dates, strings, etc.)
        exclude_columns = ['target', 'surface_hard', 'surface_clay', 'surface_grass']
        
        # Identify numeric columns only
        numeric_columns = []
        for col in training_df.columns:
            if col not in exclude_columns:
                # Check if column is numeric
                if training_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numeric_columns.append(col)
                else:
                    print(f"Skipping non-numeric column: {col} (dtype: {training_df[col].dtype})")
        
        self.feature_columns = numeric_columns
        self.X = training_df[self.feature_columns].values
        self.y = training_df['target'].values
        
        print(f"Using {len(self.feature_columns)} numeric features: {self.feature_columns}")
        
    
    def prepare_training_data(self):
        """Data is already prepared, just return it"""

        return self.X, self.y
    
    def train_and_validate(self):
        """Train logistic regression with time-series validation"""

        X, y = self.prepare_training_data()
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        brier_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred_proba = self.model.predict_proba(X_val_scaled)[:, 1]
            cv_scores.append(self.model.score(X_val_scaled, y_val))
            brier_scores.append(brier_score_loss(y_val, y_pred_proba))
        
        print(f"CV Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"CV Brier Score: {np.mean(brier_scores):.3f} ± {np.std(brier_scores):.3f}")
        
        # Final training on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        return {
            'cv_accuracy': np.mean(cv_scores),
            'cv_brier': np.mean(brier_scores),
            'feature_importance': self.model.coef_[0],  # Return coefficients directly
            'feature_names': self.feature_columns       # Include feature names
        }
    
    def predict_match_probability(self, match_features):
        """Predict probability given match features"""

        feature_array = np.array([match_features[col] for col in self.feature_columns])
        feature_array_scaled = self.scaler.transform(feature_array.reshape(1, -1))
        return self.model.predict_proba(feature_array_scaled)[0, 1]

    
    def calibration_analysis(self, X_test, y_test):
        """Analyze how well-calibrated the probabilities are"""

        X_test_scaled = self.scaler.transform(X_test)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calibration curve
        fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
        
        return {
            'calibration_slope': np.polyfit(mean_pred, fraction_pos, 1)[0],
            'calibration_data': (fraction_pos, mean_pred),
            'brier_score': brier_score_loss(y_test, y_proba)
        }