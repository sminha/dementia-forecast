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

    def load_and_preprocess_data(self, filepath):
        df = pd.read_csv(filepath)
        df = df.drop(columns=['EMAIL', 'activity_day_end'], errors='ignore')
        df = df.dropna()

        df['DIAG_NM'] = df['DIAG_NM'].replace({'CN': 0, 'MCI': 1, 'Dem': 1})

        grouped = df.groupby('ID')
        feature_data = []
        labels = []

        for _, group in grouped:
            features = group.drop(columns=['EMAIL', 'ID', 'activity_day_start', 'activity_day_end', 'DIAG_NM']).values
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

if __name__ == '__main__':
    trainer = LifelogTrainer()
    trainer.run()