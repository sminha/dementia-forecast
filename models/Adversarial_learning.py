import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lifelog_encoder import TimeSeriesDataset, TimeSeriesEncoder
from lifestyle_encoder import TableDataset, TableEncoder
import numpy as np
from config import *

# Hyperparameters
num_epochs = 6
lr = 0.00003
common_dim = 32
batch = 64

# domain discriminator for adversarial training
class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Gradient Reversal Layer for adversarial training
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=1) 



def main():

    train_ts_ds = TimeSeriesDataset(csv_path=ts_data_path, seq_len=ts_day)
    train_ts_loader = DataLoader(train_ts_ds, batch_size=batch, shuffle=True, drop_last=True)

    train_tab_ds = TableDataset(csv_path=tab_data_path)
    train_tab_loader = DataLoader(train_tab_ds, batch_size=batch, shuffle=True, drop_last=True)

    your_ts_input_dim = train_ts_ds.X.shape[2]
    your_tab_input_dim = train_tab_ds.X.shape[1]

    ts_encoder = TimeSeriesEncoder(input_size=your_ts_input_dim, hidden_size=ts_hidden_dim, bidirectional=True)
    ts_encoder.load_state_dict(torch.load('ts_encoder_subconloss.pth', map_location='cpu'))

    tab_encoder = TableEncoder(input_dim=your_tab_input_dim, hidden_dim=tab_hidden_dim, embedding_dim=tab_output_dim)
    tab_encoder.load_state_dict(torch.load('tb_encoder_subconloss.pth', map_location='cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts_encoder.to(device).train()
    tab_encoder.to(device).train()

    ts_proj = ProjectionHead(in_dim=ts_output_dim, out_dim=common_dim).to(device)
    tab_proj = ProjectionHead(in_dim=tab_output_dim, out_dim=common_dim).to(device)

    domain_discriminator = DomainDiscriminator(input_dim=common_dim).to(device)

    optimizer_feat = torch.optim.Adam(
        list(ts_encoder.parameters()) + 
        list(tab_encoder.parameters()) + 
        list(ts_proj.parameters()) + 
        list(tab_proj.parameters()),
        lr=lr
    )

    optimizer_disc = torch.optim.Adam(domain_discriminator.parameters(), lr=lr)

    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        total_feat_loss = 0.0
        total_disc_loss = 0.0

        p = epoch / num_epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        for (ts_batch, _), (tab_batch, _) in zip(train_ts_loader, train_tab_loader):
            ts_batch = ts_batch.to(device)
            tab_batch = tab_batch.to(device)
            batch_size = ts_batch.size(0)

            ts_domain_labels = torch.zeros(batch_size, 1).to(device)
            tab_domain_labels = torch.ones(batch_size, 1).to(device)

            optimizer_disc.zero_grad()
            ts_emb = ts_encoder(ts_batch)
            ts_proj_emb = ts_proj(ts_emb).detach()
            ts_domain_preds = domain_discriminator(ts_proj_emb)

            tab_emb = tab_encoder(tab_batch)
            tab_proj_emb = tab_proj(tab_emb).detach()
            tab_domain_preds = domain_discriminator(tab_proj_emb)

            disc_loss_ts = bce_loss(ts_domain_preds, ts_domain_labels)
            disc_loss_tab = bce_loss(tab_domain_preds, tab_domain_labels)
            disc_loss = disc_loss_ts + disc_loss_tab
            disc_loss.backward()
            optimizer_disc.step()
            total_disc_loss += disc_loss.item()

            optimizer_feat.zero_grad()
            ts_emb = ts_encoder(ts_batch)
            ts_proj_emb = ts_proj(ts_emb)
            ts_proj_emb_grl = GradientReversalLayer.apply(ts_proj_emb, alpha)
            ts_domain_preds = domain_discriminator(ts_proj_emb_grl)

            tab_emb = tab_encoder(tab_batch)
            tab_proj_emb = tab_proj(tab_emb)
            tab_proj_emb_grl = GradientReversalLayer.apply(tab_proj_emb, alpha)
            tab_domain_preds = domain_discriminator(tab_proj_emb_grl)

            feat_loss_ts = bce_loss(ts_domain_preds, ts_domain_labels)
            feat_loss_tab = bce_loss(tab_domain_preds, tab_domain_labels)
            feat_loss = feat_loss_ts + feat_loss_tab
            feat_loss.backward()
            optimizer_feat.step()
            total_feat_loss += feat_loss.item()

        avg_disc_loss = total_disc_loss / len(train_ts_loader)
        avg_feat_loss = total_feat_loss / len(train_ts_loader)
        print(f"에폭 {epoch+1}/{num_epochs} - 판별기 손실: {avg_disc_loss:.4f}, 특성 손실: {avg_feat_loss:.4f}")

    torch.save(ts_encoder.state_dict(), 'ts_encoder_mmd.pth')
    torch.save(tab_encoder.state_dict(), 'tab_encoder_mmd.pth')
    torch.save(ts_proj.state_dict(), 'ts_proj_adversarial.pth')
    torch.save(tab_proj.state_dict(), 'tab_proj_adversarial.pth')

    print("도메인 적응 학습 완료: 인코더와 프로젝션 헤드가 저장되었습니다.")

if __name__ == "__main__":
    main()
