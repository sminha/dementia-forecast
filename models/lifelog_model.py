import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os

class LifelogDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LifelogModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(LifelogModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

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
        df = df.drop(columns=['EMAIL', 'activity_day_end', 'activity_day_start', 'sleep_bedtime_end', 'sleep_bedtime_start'], errors='ignore')

        df = df.drop(columns = ['CONVERT(activity_class_5min USING utf8)', 'CONVERT(activity_met_1min USING utf8)', 'CONVERT(sleep_hypnogram_5min USING utf8)', 
                                'CONVERT(sleep_rmssd_5min USING utf8)', 'CONVERT(sleep_hr_5min USING utf8)'], errors='ignore')
        
        df = df.replace('...', 0)
        df = df.fillna(0)

        df['DIAG_NM'] = df['DIAG_NM'].replace({'CN': 0, 'MCI': 1, 'Dem': 1})

        grouped = df.groupby('ID')
        feature_data = []
        labels = []

        for _, group in grouped:
            features = group.drop(columns=['ID','DIAG_NM']).values
            label = group['DIAG_NM'].values[0]
            if features.shape[0] == 3:
                feature_data.append(features.flatten())
                labels.append(label)

        X = np.array(feature_data)
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
        all_labels = []
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        auc = roc_auc_score(all_labels, all_preds)
        return acc, f1, auc
    
    def save_model(self, model, input_dim, metrics=None):
        """
        모델과 관련 정보를 저장하는 메소드
        
        Args:
            model: 저장할 PyTorch 모델
            input_dim: 모델의 입력 차원
            metrics: 모델 성능 지표 (딕셔너리)
        """
        # 모델 가중치 저장
        model_path = os.path.join(self.model_save_path, "lifelog_model.pt")
        torch.save(model.state_dict(), model_path)
        
        # 모델 구조 및 하이퍼파라미터 저장
        model_info = {
            'input_dim': input_dim,
            'hidden_dim': 64,  # 모델에서 사용한 하이퍼파라미터
            'output_dim': 2,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        # 성능 지표가 제공된 경우 함께 저장
        if metrics:
            model_info.update(metrics)
        
        # 모델 정보 저장
        info_path = os.path.join(self.model_save_path, "model_info.pt")
        torch.save(model_info, info_path)
        
        print(f"모델이 저장되었습니다: {model_path}")
        print(f"모델 정보가 저장되었습니다: {info_path}")

    def run(self):
        X_train, y_train = self.load_and_preprocess_data(self.train_data_path)
        X_test, y_test = self.load_and_preprocess_data(self.test_data_path)

        train_dataset = LifelogDataset(X_train, y_train)
        test_dataset = LifelogDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model = LifelogModel(input_dim=X_train.shape[1]).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(10):
            self.train_model(model, train_loader, criterion, optimizer)

        acc, f1, auc = self.evaluate_model(model, test_loader)
        print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
        
        # 모델 저장
        metrics = {
            'accuracy': acc,
            'f1_score': f1,
            'auc': auc
        }
        self.save_model(model, X_train.shape[1], metrics)

if __name__ == '__main__':
    trainer = LifelogTrainer()
    trainer.run()
