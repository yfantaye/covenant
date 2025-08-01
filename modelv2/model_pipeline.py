import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from lifelines import CoxPHFitter
import joblib
import os
from abc import ABC, abstractmethod
from .evaluation import classification_evaluation, survival_evaluation
from .utils import calculate_z_scores, compute_alpha_beta
import logging

class BaseModelPipeline(ABC):
    """Abstract base class for a model pipeline."""
    def __init__(self, config):
        self.model_name = config['model_type']
        self.config = config
        self.params = self.config['models'].get(self.model_name, {})
        self.model = None
        self.scaler = StandardScaler()
        self.output_dir = f"{self.config.get("model_output_directory","model_outputs")}/{self.model_name}"
        os.makedirs(self.output_dir, exist_ok=True)

    def _split_data(self, df: pd.DataFrame):
        """Time-based split of the data."""
        # A proper time-based split is crucial. Here's a simple version.
        df = df.sort_values('signal_date')
        split_date = df['signal_date'].quantile(0.8, interpolation='nearest')
        
        train_df = df[df['signal_date'] < split_date]
        test_df = df[df['signal_date'] >= split_date]

        self.X_train = train_df.drop(columns=['companyid', 'signal_date', 'target'])
        self.y_train = train_df['target']
        self.X_test = test_df.drop(columns=['companyid', 'signal_date', 'target'])
        self.y_test = test_df['target']

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def _predict(self):
        pass

    @abstractmethod
    def _evaluate(self):
        pass

    def run(self, df: pd.DataFrame):
        """Runs the full train, predict, evaluate pipeline."""
        self._split_data(df)
        self._train()
        self.predictions = self._predict()
        self.evaluation_results = self._evaluate()
        return self.evaluation_results

    def save_artifacts(self):
        """Saves the trained model and evaluation results."""
        if hasattr(self, 'model') and self.model is not None:
            joblib.dump(self.model, os.path.join(self.output_dir, "model.joblib"))
        if hasattr(self, 'scaler') and self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(self.output_dir, "scaler.joblib"))
        if hasattr(self, 'evaluation_results') and self.evaluation_results is not None:
            if 'plots' in self.evaluation_results and 'roc_curve' in self.evaluation_results['plots']:
                self.evaluation_results['plots']['roc_curve'].savefig(os.path.join(self.output_dir, "roc_curve.png"))
            if 'metrics' in self.evaluation_results:
                pd.DataFrame(self.evaluation_results['metrics'], index=[0]).to_csv(os.path.join(self.output_dir, "metrics.csv"))

class ZScorePipeline(BaseModelPipeline):
    def __init__(self, params=None):
        super().__init__("z_score", params)
        self.params = params or {'z_score_threshold': 1.0}
        self.weights = None

    def _train(self):
        """Compute feature weights using Z-score analysis."""
        # Combine train data for weight calculation
        train_data = pd.concat([self.X_train, self.y_train], axis=1)
        
        logging.info(f"Training data shape: {train_data.shape}")
        logging.info(f"Target column name: {self.y_train.name}")
        logging.info(f"Feature columns: {list(self.X_train.columns)}")
        
        # Calculate Z-scores for feature importance
        z_scores_df = calculate_z_scores(train_data, 'target')
        
        # Store weights as a Series indexed by feature names
        self.weights = pd.Series(z_scores_df['Score'].values, index=z_scores_df['Feature'])
        
        # Log the top features
        top_features = z_scores_df.head(10)
        logging.info(f"Top 10 most sensitive features:")
        for _, row in top_features.iterrows():
            logging.info(f"  {row['Feature']}: {row['Score']:.4f}")

    def _predict(self):
        """Calculate scores using the computed weights."""
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        logging.info(f"Test data shape: {self.X_test.shape}")
        logging.info(f"Available features in test: {list(self.X_test.columns)}")
        
        # Calculate scores using dot product of features and weights
        # Ensure columns are in the same order as weights
        feature_cols = self.weights.index
        available_cols = [col for col in feature_cols if col in self.X_test.columns]
        
        logging.info(f"Features with weights: {len(feature_cols)}")
        logging.info(f"Available features in test: {len(available_cols)}")
        
        if len(available_cols) == 0:
            raise ValueError("No matching features found between training and test data")
        
        # Use only available features and align with weights
        weights_subset = self.weights[available_cols]
        X_test_subset = self.X_test[available_cols]
        
        # Calculate scores
        scores = X_test_subset.dot(weights_subset)
        
        logging.info(f"Score statistics: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        # Normalize scores to 0-1 range for evaluation
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = scores
        
        return normalized_scores

    def _evaluate(self):
        """Evaluate the model using classification metrics."""
        return classification_evaluation(self.y_test, self.predictions)

    def save_artifacts(self):
        """Save the trained model artifacts including weights."""
        # Save weights
        if self.weights is not None:
            weights_df = pd.DataFrame({
                'feature': self.weights.index,
                'weight': self.weights.values
            })
            weights_df.to_csv(os.path.join(self.output_dir, "feature_weights.csv"), index=False)
            logging.info(f"Feature weights saved to {self.output_dir}/feature_weights.csv")
        
        # Call parent method for evaluation results
        super().save_artifacts()

class LogisticRegressionPipeline(BaseModelPipeline):
    def __init__(self, params=None):
        super().__init__("logistic_regression", params)
        self.params = params or {'C': 1.0, 'penalty': 'l1', 'solver': 'liblinear', 'class_weight': 'balanced'}

    def _train(self):
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.model = LogisticRegression(**self.params, random_state=42)
        self.model.fit(self.X_train_scaled, self.y_train)

    def _predict(self):
        X_test_scaled = self.scaler.transform(self.X_test)
        return self.model.predict_proba(X_test_scaled)[:, 1]

    def _evaluate(self):
        return classification_evaluation(self.y_test, self.predictions)

class LightGBMPipeline(BaseModelPipeline):
    def __init__(self, params=None):
        super().__init__("lightgbm", params)
        self.params = params or {'objective': 'binary', 
                                 'metric': 'auc', 
                                 'n_estimators': 1000, 
                                 'learning_rate': 0.05,
                                 'num_leaves': 31,
                                 'max_depth': -1,
                                 'random_state': 42,
                                 'n_jobs': -1,
        }

    def _train(self):
        self.model = lgb.LGBMClassifier(**self.params, random_state=42)
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            eval_metric=self.params.get('metric', 'auc'),
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )

    def _predict(self):
        return self.model.predict_proba(self.X_test)[:, 1]

    def _evaluate(self):
        return classification_evaluation(self.y_test, self.predictions)

class CoxPHPipeline(BaseModelPipeline):
    def __init__(self, params=None):
        super().__init__("coxph", params)
        self.params = params or {'penalizer': 0.1}

    def _split_data(self, df: pd.DataFrame):
        # CoxPH needs a different data structure
        # This is a placeholder; feature engineering must create these columns
        if 'duration' not in df.columns or 'event' not in df.columns:
             raise NotImplementedError("CoxPH needs 'duration' and 'event' columns created in feature engineering.")
        
        df = df.sort_values('signal_date')
        split_date = df['signal_date'].quantile(0.8, interpolation='nearest')
        
        self.train_df = df[df['signal_date'] < split_date]
        self.test_df = df[df['signal_date'] >= split_date]

    def _train(self):
        self.model = CoxPHFitter(**self.params)
        self.model.fit(self.train_df, duration_col='duration', event_col='event')

    def _predict(self):
        # Predicts partial hazard, not probability
        return self.model.predict_partial_hazard(self.test_df)

    def _evaluate(self):
        # Uses survival-specific metrics
        return survival_evaluation(self.model, self.test_df)
