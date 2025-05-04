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
        self.lifelog_model_path = os.path.join(self.model_dir, "lifelog_model2.pt")
        self.lifelog_info_path = os.path.join(self.model_dir, "model_info2.pt")
        self.lifestyle_model_path = os.path.join(self.model_dir, "lifestyle_model2.pkl")
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

        # 저장된 스케일러 및 라벨 인코더 불러오기
        self.scaler = model_info['scaler']
        self.label_encoder = model_info['label_encoder']
        self.lifestyle_model.load_model(self.lifestyle_model_path)

        print("Models successfully loaded.")

    def predict(self, lifelog_data, lifestyle_data):
        # lifelog_data가 numpy 배열이 아니라면 변환
        if isinstance(lifelog_data, pd.DataFrame):
            X_lifelog = lifelog_data.values
        elif isinstance(lifelog_data, np.ndarray):
            X_lifelog = lifelog_data
        else:
            raise ValueError("lifelog_data는 DataFrame 또는 numpy.ndarray 형식이어야 합니다.")

        lifelog_tensor = torch.tensor(X_lifelog, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            lifelog_outputs = self.lifelog_model(lifelog_tensor)
            lifelog_probs = torch.softmax(lifelog_outputs, dim=1)[:, 1].cpu().numpy()

        # lifestyle_data도 필요시 numpy로 변환
        if isinstance(lifestyle_data, pd.DataFrame):
            X_lifestyle = lifestyle_data
        else:
            raise ValueError("lifestyle_data는 pandas DataFrame 형식이어야 합니다.")

        lifestyle_probs = self.lifestyle_model.predict(X_lifestyle)

        # 앙상블 예측
        ensemble_pred = (
            self.ensemble_weights["lifelog"] * lifelog_probs +
            self.ensemble_weights["lifestyle"] * lifestyle_probs
        )
        return ensemble_pred
    
    def preprocess_lifelog_data(self, filepath):
        print(f"Loading lifelog data from {filepath}...")
        df = pd.read_csv(filepath)
        df = df[['activity_cal_active', 'activity_cal_total', 'activity_daily_movement', 'activity_day_end', 
                'activity_day_start', 'activity_high', 'activity_inactive', 'activity_medium', 
                'activity_met_1min', 'activity_met_min_high', 'activity_met_min_inactive', 'activity_met_min_low', 
                'activity_met_min_medium', 'activity_non_wear', 'activity_steps', 'activity_total', 'sleep_awake', 
                'sleep_bedtime_end', 'sleep_bedtime_start', 'sleep_deep', 'sleep_duration', 'sleep_efficiency', 
                'sleep_hypnogram_5min', 'sleep_is_longest', 'sleep_light', 'sleep_midpoint_at_delta', 'sleep_midpoint_time', 
                'sleep_period_id', 'sleep_rem', 'sleep_rmssd', 'sleep_rmssd_5min', 'sleep_total',
                'ID', 'DIAG_NM']].copy()

        str_columns_to_encode = [col for col in df.columns if df[col].dtype == 'object' and col != 'DIAG_NM']
        for col in str_columns_to_encode:
            df[col] = self.label_encoder.transform(df[col])  

        df = df.fillna(0)
        df['DIAG_NM'] = df['DIAG_NM'].replace({'CN': 0, 'MCI': 1, 'Dem': 1})


        feature_data = []
        labels = []

        for i in range(0, len(df) - (self.temp - 1), self.temp):  
            window = df.iloc[i:i + self.temp]
            features = window.drop(columns=['ID', 'DIAG_NM']).values
            label = window['DIAG_NM'].values[0]

            feature_data.append(features.flatten())
            labels.append(label)

        X = np.array(feature_data, dtype=np.float32)
        y = np.array(labels)
        X = self.scaler.transform(X)
        y = self.label_encoder.transform(y)

        return X, y
    
    def set_ensemble_weights(self, lifelog_weight=0.5, lifestyle_weight=0.5):
        total = lifelog_weight + lifestyle_weight
        self.ensemble_weights = {
            "lifelog": lifelog_weight / total,
            "lifestyle": lifestyle_weight / total
        }
        print(f"Ensemble weights set: Lifelog={self.ensemble_weights['lifelog']:.2f}, Lifestyle={self.ensemble_weights['lifestyle']:.2f}")
    
    def evaluate(self, lifelog_test_path, lifestyle_test_path):
        X_lifelog, _ = self.preprocess_lifelog_data(lifelog_test_path)
        lifestyle_test = pd.read_csv(lifestyle_test_path)
        X_lifestyle = lifestyle_test[self.lifestyle_model.feature_cols]
        y_lifestyle = lifestyle_test[self.lifestyle_model.target_col]
        
        predictions = self.predict(X_lifelog, X_lifestyle)
        auc_score = roc_auc_score(y_lifestyle, predictions)
        print(f"Multimodal model ROC AUC score: {auc_score:.4f}")
        return auc_score
    
    def run(self, lifelog_test_path="data/lifelog_test.csv", lifestyle_test_path="data/lifestyle_test.csv"):
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
