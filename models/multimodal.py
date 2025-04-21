import pandas as pd
import numpy as np
import torch
import os
import joblib
from sklearn.metrics import roc_auc_score
from .lifelog_model import LifelogModel, LifelogTrainer
from .lifestyle_model import Tabular_Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.serialization

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

class MultimodalModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = os.path.join(BASE_DIR, "models")
        self.lifelog_trainer = LifelogTrainer()
        self.lifestyle_model = Tabular_Model(data_dir="../data")
        self.lifelog_model_path = os.path.join(self.model_dir, "lifelog_model.pt")
        self.lifelog_info_path = os.path.join(self.model_dir, "model_info.pt")
        self.lifestyle_model_path = os.path.join(self.model_dir, "lifestyle_model.pkl")
        self.lifelog_model = None
        self.ensemble_weights = {"lifelog": 0.5, "lifestyle": 0.5}

    def load_models(self):
        torch.serialization.add_safe_globals([
            StandardScaler,
            LabelEncoder,
            np.core.multiarray.scalar 
        ])

        model_info = torch.load(self.lifelog_info_path, weights_only=False)

        self.lifelog_model = LifelogModel(input_dim=model_info['input_dim']).to(self.device)
        self.lifelog_model.load_state_dict(torch.load(self.lifelog_model_path, weights_only=False))
        self.lifelog_model.eval()

        self.lifestyle_model.load_model(self.lifestyle_model_path)

        print("Models successfully loaded.")

    def predict(self, lifelog_data, lifestyle_data):
        # lifelog feature vector를 모두 0으로 만듦
        NUMERIC_INDICES_ONE_DAY = (
            list(range(1, 32))
        )
        days = 3  # 또는 실제 들어온 days 수
        n_features = len(NUMERIC_INDICES_ONE_DAY) * days
        X_lifelog = np.zeros((1, n_features), dtype=float)
        lifelog_tensor = torch.tensor(X_lifelog, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            lifelog_outputs = self.lifelog_model(lifelog_tensor)
            lifelog_probs = torch.softmax(lifelog_outputs, dim=1)[:, 1].cpu().numpy()

        lifestyle_probs = self.lifestyle_model.predict(lifestyle_data)
        ensemble_pred = (
            self.ensemble_weights["lifelog"] * lifelog_probs +
            self.ensemble_weights["lifestyle"] * lifestyle_probs
        )
        return ensemble_pred

    
    def set_ensemble_weights(self, lifelog_weight=0.5, lifestyle_weight=0.5):
        total = lifelog_weight + lifestyle_weight
        self.ensemble_weights = {
            "lifelog": lifelog_weight / total,
            "lifestyle": lifestyle_weight / total
        }
        print(f"Ensemble weights set: Lifelog={self.ensemble_weights['lifelog']:.2f}, Lifestyle={self.ensemble_weights['lifestyle']:.2f}")
    
    def evaluate(self, lifelog_test_path, lifestyle_test_path):
        X_lifelog, y_true = self.lifelog_trainer.load_and_preprocess_data(lifelog_test_path)
        lifestyle_test = pd.read_csv(lifestyle_test_path)
        X_lifestyle = lifestyle_test[self.lifestyle_model.feature_cols]
        y_lifestyle = lifestyle_test[self.lifestyle_model.target_col]
        
        predictions = self.predict(X_lifelog, X_lifestyle)
        auc_score = roc_auc_score(y_lifestyle, predictions)
        print(f"Multimodal model ROC AUC score: {auc_score:.4f}")
        return auc_score
    
    def run(self, lifelog_test_path="../data/lifelog_test.csv", lifestyle_test_path="../data/lifestyle_test.csv"):
        if self.lifelog_model is None:
            print("Loading models...")
            self.load_models()
        
        print("Processing test data...")
        X_lifelog, _ = self.lifelog_trainer.load_and_preprocess_data(lifelog_test_path)
        lifestyle_test = pd.read_csv(lifestyle_test_path)
        X_lifestyle = lifestyle_test[self.lifestyle_model.feature_cols]
        
        print("Generating predictions...")
        predictions = self.predict(X_lifelog, X_lifestyle)
        
        if self.lifestyle_model.target_col in lifestyle_test.columns:
            y_true = lifestyle_test[self.lifestyle_model.target_col]
            auc_score = roc_auc_score(y_true, predictions)
            print(f"Prediction complete. Model performance: ROC AUC = {auc_score:.4f}")
        else:
            print("Prediction complete.")
        
        return predictions

if __name__ == "__main__":
    model = MultimodalModel()
    predictions = model.run()
    print(f"Prediction: {predictions.shape}")
