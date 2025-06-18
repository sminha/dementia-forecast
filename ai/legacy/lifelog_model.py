import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import os
import random

temp = 3
class LifelogDataset(Dataset):
    def __init__(self, data, labels, seq_len=1):
        self.data = torch.tensor(data, dtype=torch.float32).reshape(-1, seq_len, data.shape[1] // seq_len)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class LifelogModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, num_layers=1, bidirectional=False):
        super(LifelogModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                             hidden_size=hidden_dim, 
                             num_layers=num_layers, 
                             batch_first=True, 
                             bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        if lstm_out.dim() == 2:
            lstm_out = lstm_out.unsqueeze(1) 
        
        if lstm_out.dim() != 3:
            raise ValueError(f"LSTM output should be a 3D tensor, but got {lstm_out.dim()} dimensions")
        
        lstm_out = lstm_out[:, -1, :]  
        out = self.fc(lstm_out)
        return out
    
class LifelogTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.train_data_path = os.path.join("..", "data", "lifelog_train.csv")
        self.test_data_path = os.path.join("..", "data", "lifelog_test.csv")
        self.model_save_path = os.path.join("..", "models")
        
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    def load_and_preprocess_data(self, filepath):
        df = pd.read_csv(filepath)
        df = df[['activity_cal_active', 'activity_cal_total', 'activity_daily_movement', 'activity_day_end', 
                'activity_day_start', 'activity_high', 'activity_inactive', 'activity_medium', 
                'activity_met_1min', 'activity_met_min_high', 'activity_met_min_inactive', 'activity_met_min_low', 
                'activity_met_min_medium', 'activity_non_wear', 'activity_steps', 'activity_total', 'sleep_awake', 
                'sleep_bedtime_end', 'sleep_bedtime_start', 'sleep_deep', 'sleep_duration', 'sleep_efficiency', 
                'sleep_hypnogram_5min', 'sleep_is_longest', 'sleep_light', 'sleep_midpoint_at_delta', 'sleep_midpoint_time', 
                'sleep_period_id', 'sleep_rem', 'sleep_rmssd', 'sleep_rmssd_5min', 'sleep_total',
                'ID', 'DIAG_NM']].copy()

        str_columns_to_encode = [col for col in df.columns 
                                if df[col].dtype == 'object' and col != 'DIAG_NM']
        
        for col in str_columns_to_encode:
            df[col] = self.label_encoder.fit_transform(df[col])

        df = df.fillna(0)
        print(df.shape)
        df['DIAG_NM'] = df['DIAG_NM'].replace({'CN': 0, 'MCI': 1, 'Dem': 1})

        feature_data = []
        labels = []
        
        for i in range(0, len(df) - (temp-1), temp):  
            window = df.iloc[i:i+temp]
            features = window.drop(columns=['ID', 'DIAG_NM']).values
            label = window['DIAG_NM'].values[0] 
            
            feature_data.append(features.flatten())
            labels.append(label)

        X = np.array(feature_data, dtype=np.float32)
        y = np.array(labels)

        X = self.scaler.fit_transform(X)
        y = self.label_encoder.fit_transform(y)
        return X, y
    
    def train_model(self, model, train_loader, criterion, optimizer):
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def evaluate_model(self, model, test_loader):
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)  
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
        return acc, f1, auc
    
    def save_model(self, model, input_dim):
        model_path = os.path.join(self.model_save_path, "lifelog_model2.pt")
        torch.save(model.state_dict(), model_path)

        model_info = {
            'input_dim': input_dim,
            'hidden_dim': 64,  
            'output_dim': 2,  
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        info_path = os.path.join(self.model_save_path, "model_info2.pt")
        torch.save(model_info, info_path)
        
        print(f"모델이 저장되었습니다: {model_path}")
        print(f"모델 정보가 저장되었습니다: {info_path}")

    def run(self):
        def set_seed(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        set_seed(42)

        X_train, y_train = self.load_and_preprocess_data(self.train_data_path)
        X_test, y_test = self.load_and_preprocess_data(self.test_data_path)

        train_dataset = LifelogDataset(X_train, y_train)
        test_dataset = LifelogDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size = 64, shuffle=False)

        input_dim = X_train.shape[1]
        model = LifelogModel(input_dim=input_dim).to(self.device)
        from collections import Counter
        class_counts = Counter(y_train)
        total = sum(class_counts.values())
        class_weights = [total / class_counts[i] for i in sorted(class_counts)]
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)

        for epoch in range(100):
            self.train_model(model, train_loader, criterion, optimizer)
            test_acc, test_f1, test_auc = self.evaluate_model(model, test_loader)
            print(f"[Epoch {epoch}] Test  - Accuracy: {test_acc:.4f}, F1 Score: {test_f1:.4f}, AUC: {test_auc:.4f}")
                
        
        self.save_model(model, X_train.shape[1])

if __name__ == '__main__':
    trainer = LifelogTrainer()
    trainer.run()
