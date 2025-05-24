import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import joblib
import os
from torch.utils.data import Dataset, DataLoader

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.output_dim = hidden_size * (2 if bidirectional else 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        if self.lstm.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1]
        return F.normalize(h, dim=1)
    
class TableEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.output_dim = embedding_dim

    def forward(self, x):
        return self.net(x)

class JointClassifier(nn.Module):
    def __init__(self, enc_time, enc_tab, hidden_dim, num_classes):
        super().__init__()
        self.enc_time = enc_time
        self.enc_tab = enc_tab
        total_dim = enc_time.output_dim + enc_tab.output_dim
        self.fc = nn.Linear(total_dim, num_classes)

    def forward(self, x_ts, x_tab):
        ts_out = self.enc_time(x_ts)
        tab_out = self.enc_tab(x_tab)
        combined = torch.cat([ts_out, tab_out], dim=1)
        return self.fc(combined)

class MultimodalModel:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.embedding_dim = 8
        self.ts_hidden_dim = 32
        self.tab_hidden_dim = 32
        self.ts_output_dim = 64
        self.tab_output_dim = 32
        self.total_dim = self.ts_output_dim + self.tab_output_dim
        self.ts_day = 3
        self.batch = 64
        self.projection_dim = 16
        self.joint_hidden_dim = 32
        
        self.seq_len = self.ts_day
        self.ts_features = 28
        
        self.lifestyle_scaler = None
        self.lifelog_scaler = None
        self.models_dir = os.path.join(os.getcwd(), 'models')
        
    def load_models(self):
        try:
            ts_input_dim = self.ts_features 
            tab_input_dim = 17
            
            enc_time = TimeSeriesEncoder(input_size=ts_input_dim, 
                                       hidden_size=self.ts_hidden_dim)
            enc_tab = TableEncoder(input_dim=tab_input_dim,
                                 hidden_dim=self.tab_hidden_dim,
                                 embedding_dim=self.tab_output_dim)
            
            self.model = JointClassifier(enc_time, enc_tab,
                                       hidden_dim=self.joint_hidden_dim,
                                       num_classes=2)
            
            enc_time.load_state_dict(torch.load(os.path.join(self.models_dir, 'ts_encoder_contrastive.pt'), map_location='cpu'))
            enc_tab.load_state_dict(torch.load(os.path.join(self.models_dir, 'tab_encoder_contrastive.pt'), map_location='cpu'))
            self.model.fc.load_state_dict(torch.load(os.path.join(self.models_dir, 'joint_classifier_fc.pt'), map_location='cpu'))
            
            self.model.to(self.device)
            self.model.eval()
            
            self.lifestyle_scaler = joblib.load(os.path.join(self.models_dir, 'lifestyle_scaler.pkl'))
            self.lifelog_scaler = joblib.load(os.path.join(self.models_dir, 'lifelog_scaler.pkl'))
            
            print("model and scalers loaded successfully.")
            
        except Exception as e:
            print(f"Failed to load model and scalers: {e}")
            raise e
    
    def predict(self, lifelog_data, lifestyle_data):
        try:
            with torch.no_grad():
                lifestyle_scaled = self.lifestyle_scaler.transform([lifestyle_data])
                
                lifelog_array = np.array(lifelog_data).reshape(self.seq_len, self.ts_features)
                lifelog_scaled = self.lifelog_scaler.transform(lifelog_array)
                lifelog_scaled = lifelog_scaled.reshape(1, self.seq_len, self.ts_features)
                
                x_ts = torch.FloatTensor(lifelog_scaled).to(self.device)
                x_tab = torch.FloatTensor(lifestyle_scaled).to(self.device)
                
                logits = self.model(x_ts, x_tab)
                probs = torch.softmax(logits, dim=1)
                risk_score = probs[0, 1].cpu().numpy()
                risk_score = round(float(risk_score), 4)
                
                return risk_score
                
        except Exception as e:
            print(f"prediction failure: {e}")
            raise e


def main():

    with open('test_input_sample.json', 'r') as f:
        test_data = json.load(f)
    
    print(f"Lifelog 데이터 길이: {len(test_data['lifelog'])}")
    print(f"Lifestyle 데이터 길이: {len(test_data['lifestyle'])}")
    
    model = MultimodalModel()
    model.load_models()
    
    print("\n=== 예측 수행 ===")
    risk_score = model.predict(test_data['lifelog'], test_data['lifestyle'])
    is_dementia = risk_score > 0.5
    risk_score = round(risk_score, 4)
    print(f"위험도 점수: {risk_score:.4f}")
    print(f"치매 여부: {is_dementia}")
    print(f"분류 결과: {'치매 위험' if is_dementia else '정상'}")
    

if __name__ == "__main__":
    main()
