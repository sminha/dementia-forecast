import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import joblib
from config import *


#hyperparameters
epochs = 10
learning_rate = 0.001

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, seq_len):
        df = pd.read_csv(csv_path)
        df['DIAG_NM'] = df['DIAG_NM'].replace({'CN': 0, 'MCI': 1, 'Dem': 1})

        scaler_path = 'lifelog_scaler.pkl'
        self.scaler = joblib.load(scaler_path)

        grouped = df.groupby('ID')
        sequences, labels = [], []

        for _id, grp in grouped:
            feats = grp.drop(columns=['ID', 'Day', 'DIAG_NM', 'EMAIL'], errors='ignore')
            feats = feats.select_dtypes(include=[np.number])

            arr = feats.values.astype(np.float32)
            arr = self.scaler.transform(arr)

            if arr.shape[0] != seq_len:
                continue

            sequences.append(arr)
            labels.append(grp['DIAG_NM'].iloc[0])

        self.X = torch.from_numpy(np.stack(sequences))
        self.y = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=ts_hidden_dim, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
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

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        loss = -mean_log_prob_pos.mean()
        return loss

def main():
    train_csv = tab_data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = TimeSeriesDataset(csv_path=train_csv, seq_len=ts_day)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)

    input_size = train_ds.X.shape[2]
    encoder = TimeSeriesEncoder(input_size=input_size).to(device)

    criterion = SupConLoss()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    encoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            features = encoder(X_batch)
            loss = criterion(features, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_ds)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    torch.save(encoder.state_dict(), 'ts_encoder_subconloss.pth')
    print("Encoder pretrained and saved.")

if __name__ == "__main__":
    main()
