import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib

class Tabular_Model:
    def __init__(self, data_dir, target_col='치매여부_치매1기타0', exclude_cols=None):
        self.data_dir = data_dir
        self.target_col = target_col
        self.exclude_cols = exclude_cols if exclude_cols else ['ID', target_col]
        self.feature_cols = None  # Will be set when loading data
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
    
    def load_data(self):
        df = pd.read_csv(os.path.join("..", "data", "lifestyle_train.csv"))
        # Set feature columns dynamically (all columns except excluded ones)
        self.feature_cols = [col for col in df.columns if col not in self.exclude_cols]
        X = df[self.feature_cols]
        y = df[self.target_col]
        return X, y

    def load_test_data(self):
        df = pd.read_csv(os.path.join("..", "data", "lifestyle_test.csv"))
        X_test = df[self.feature_cols]
        y_test = df[self.target_col]
        return X_test, y_test

    def preprocess_data(self, X):
        return self.scaler.fit_transform(X)
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        }
        
        model = XGBClassifier(**param)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        return score
    
    def train(self):
        X, y = self.load_data()
        X = self.preprocess_data(X)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, self.X_train, self.y_train, self.X_val, self.y_val), n_trials=50)
        self.best_params = study.best_params
        print("Best parameters: ", self.best_params)

        self.model = XGBClassifier(**self.best_params)
        self.model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
    
    def predict(self, X_new):
        X_new = self.scaler.transform(X_new)
        return self.model.predict_proba(X_new)[:, 1]
    
    def evaluate(self):
        X_test, y_test = self.load_test_data()
        X_test = self.scaler.transform(X_test)
        y_pred = self.model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        print(f"ROC AUC Score on test set: {auc_score}")
        return auc_score
    
    def save_model(self, model_path):
        if self.model is None:
            print("No model to save. Please train the model first.")
            return
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'best_params': self.best_params,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.best_params = model_data['best_params']
        self.feature_cols = model_data['feature_cols']
        self.target_col = model_data['target_col']
        
        print(f"Model loaded from {model_path}")
    
    def run(self, save_path):
        self.train()
        auc_score = self.evaluate()
        self.save_model(save_path)
            
        return auc_score


# Example usage in main
if __name__ == "__main__":
    # Initialize the model with the target column '치매여부_치매1기타0'
    # and exclude 'ID' and the target column from features
    model = Tabular_Model(
        data_dir="./data",
        target_col="치매여부_치매1기타0",
        exclude_cols=["ID", "치매여부_치매1기타0"]
    )
    
    # Run the model pipeline
    # To train a new model and save it:
    auc_score = model.run(save_path="./lifestyle_model.pkl")
    
    # To load a pre-trained model and evaluate it:
    print(f"Final ROC AUC Score: {auc_score}")
