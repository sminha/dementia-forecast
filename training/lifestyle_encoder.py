import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import joblib
from config import *

#hyperparameters
temperature    = 0.05
epochs         = 10
learning_rate = 0.001

class TableDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.y = torch.tensor(df['치매여부_치매1기타0'].values, dtype=torch.long)

        X = df.drop(columns=['ID', '치매여부_치매1기타0'], errors='ignore')
        X = X.astype(np.float32)
        scaler = joblib.load('lifestyle_scaler.pkl')
        X = scaler.transform(X)
        self.X = torch.from_numpy(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

def supervised_contrastive_loss(features, labels):
    device = features.device
    batch_size = features.size(0)

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    mask_offdiag = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * mask_offdiag

    exp_logits = torch.exp(logits) * mask_offdiag
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    return - mean_log_prob_pos.mean()

if __name__ == "__main__":
    batch_size = batch
    train_csv = ts_data_path

    table_ds    = TableDataset(train_csv)
    table_loader= DataLoader(table_ds, batch_size=batch_size, shuffle=True)

    input_dim = table_ds.X.shape[1]
    encoder   = TableEncoder(input_dim=input_dim,
                             hidden_dim=tab_hidden_dim,
                             embedding_dim=tab_output_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    encoder.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for X_batch, y_batch in table_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            embeddings = encoder(X_batch)  
            embeddings = nn.functional.normalize(embeddings, dim=1)
            loss = supervised_contrastive_loss(embeddings, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(table_ds)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    torch.save(encoder.state_dict(), 'tb_encoder_subconloss.pth')
    print("Encoder pretrained and saved.")
