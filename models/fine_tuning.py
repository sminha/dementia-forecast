import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import joblib
from config import *
from lifelog_encoder import TimeSeriesEncoder
from lifestyle_encoder import TableEncoder

# Hyperparameters
num_epochs_ft   = 7
lr_ft           = 0.001
reduce_class1_ratio=0.75

# compute class weights for imbalanced dataset
def compute_class_weights(dataset, reduce_class1_ratio):
    y = dataset.y.numpy()
    counter = Counter(y)
    total = sum(counter.values())
    num_classes = len(counter)
    weights = [total / counter[i] for i in range(num_classes)]
    if 1 in counter:
        weights[1] *= reduce_class1_ratio
    return torch.tensor(weights, dtype=torch.float)

# create a joint dataset combining time series and tabular data
class JointDataset(Dataset):
    def __init__(self, ts_csv, tab_csv, seq_len=ts_day):
        # 시계열
        df_ts = pd.read_csv(ts_csv)
        df_ts['DIAG_NM'] = df_ts['DIAG_NM'].replace({'CN':0, 'MCI':1, 'Dem':1})
        seqs, labels, ids = [], [], []
        for _id, grp in df_ts.groupby('ID'):
            if 'Day' in grp.columns:
                grp = grp.sort_values('Day')
            feats = grp.drop(columns=['ID','Day','DIAG_NM','EMAIL'], errors='ignore')
            feats = feats.select_dtypes(include=[np.number]).values.astype(np.float32)
            if feats.shape[0] != seq_len:
                continue
            seqs.append(feats)
            labels.append(grp['DIAG_NM'].iloc[0])
            ids.append(_id)
        self.X_ts = torch.tensor(np.stack(seqs))            
        self.y    = torch.tensor(labels, dtype=torch.long)   
        self.ids  = ids

 
        df_tab = pd.read_csv(tab_csv).set_index('ID')
        X_tab = df_tab.drop(columns=['치매여부_치매1기타0','ID'], errors='ignore')
        X_tab = X_tab.astype(np.float32)
        scaler = joblib.load('lifestyle_scaler.pkl')
        X_tab = scaler.transform(X_tab.values)

        self.X_tab_dict = {
            idx: torch.tensor(row, dtype=torch.float32)
            for idx, row in zip(df_tab.index, X_tab)
        }

    def __len__(self):
        return len(self.X_ts)

    def __getitem__(self, i):
        x_ts = self.X_ts[i]
        _id  = self.ids[i]
        x_tab= self.X_tab_dict[_id]
        y    = self.y[i]
        return x_ts, x_tab, y

# fine-tuning funcction
def finetune(model, loader, device, epochs, lr):
    weights = compute_class_weights(loader.dataset, reduce_class1_ratio=reduce_class1_ratio).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for ep in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_ts, x_tab, y in loader:
            x_ts  = x_ts.to(device)
            x_tab = x_tab.to(device)
            y     = y.to(device)

            logits = model(x_ts, x_tab)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

        avg_loss = total_loss / total
        acc      = correct / total
        print(f"[Epoch {ep}/{epochs}] Loss: {avg_loss:.4f}, Acc: {acc:.4f}")

# Joint Classifier combining time series and tabular encoders
class JointClassifier(nn.Module):
    def __init__(self, enc_time, enc_tab, num_classes=2):
        super().__init__()
        self.enc_time = enc_time
        self.enc_tab  = enc_tab
        total_dim     = enc_time.output_dim + enc_tab.output_dim
        self.fc       = nn.Linear(total_dim, num_classes)

    def forward(self, x_ts, x_tab):
        ts_out  = self.enc_time(x_ts)   # (B, ts_output_dim)
        tab_out = self.enc_tab(x_tab)   # (B, tab_output_dim)
        combined= torch.cat([ts_out, tab_out], dim=1)
        return self.fc(combined)

def main():
    ts_csv  = ts_data_path
    tab_csv = tab_data_path

    joint_ds = JointDataset(ts_csv, tab_csv, seq_len=ts_day)
    loader   = DataLoader(joint_ds, batch_size=batch, shuffle=True)

    ts_in_dim  = joint_ds.X_ts.shape[2]
    tab_in_dim = joint_ds.X_tab_dict[joint_ds.ids[0]].shape[0]

    enc_time = TimeSeriesEncoder(input_size=ts_in_dim,
                                 hidden_size=ts_hidden_dim,
                                 bidirectional=True)
    enc_tab  = TableEncoder(input_dim=tab_in_dim,
                            hidden_dim=tab_hidden_dim,
                            embedding_dim=tab_output_dim)

    enc_time.load_state_dict(torch.load('ts_encoder_mmd.pth', map_location='cpu'))
    enc_tab.load_state_dict(torch.load('tab_encoder_mmd.pth', map_location='cpu'))

    model = JointClassifier(enc_time, enc_tab, num_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finetune(model, loader, device, epochs=num_epochs_ft, lr=lr_ft)

    torch.save(enc_time.state_dict(), 'ts_encoder_contrastive.pt')
    torch.save(enc_tab.state_dict(),  'tab_encoder_contrastive.pt')
    torch.save(model.fc.state_dict(), 'joint_classifier_fc.pt')
    print("All models saved.")

if __name__ == "__main__":
    main()
