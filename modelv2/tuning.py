import optuna
from sklearn.model_selection import TimeSeriesSplit

class OptunaTuner:
    def __init__(self, model_pipeline_class, df):
        self.model_pipeline_class = model_pipeline_class
        self.df = df

    def _objective(self, trial):
        # Define hyperparameter search space
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
        }
        
        # Using a simple time split for tuning demonstration
        # A more robust approach would use TimeSeriesSplit cross-validation
        pipeline = self.model_pipeline_class(params=params)
        results = pipeline.run(self.df)
        
        return results['metrics']['roc_auc']

    def tune(self, n_trials=50):
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=n_trials)
        return study.best_params
