import os
from typing import Literal
import datetime
import time

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter


class DTIDataset(Dataset):
    def __init__(self, root: str = "ic-50-prediction/processed",
                 phase: Literal["train", "val", "test"] = "train"):
        path = os.path.join(root, phase)
        self.phase = phase
        self.data = pd.read_csv(os.path.join(path, "data.csv"))
        self.mol_embeds = pd.read_csv(os.path.join(path, "mol_embeds.csv"))
        self.mol_embeds = self.mol_embeds.set_index(['Ligand SMILES'])
        self.seq_embeds = pd.read_csv(os.path.join(path, "seq_embeds.csv"))
        self.seq_embeds = self.seq_embeds.set_index(['BindingDB Target Chain Sequence'])
        self.indices = self.data.index.values

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        sample = self.data.iloc[self.indices[idx]]

        mol_embeds = self.mol_embeds.loc[sample['Ligand SMILES']]['mol_embeds']
        seq_embeds = self.seq_embeds.loc[sample['BindingDB Target Chain Sequence']]['seq_embeds']
        ic50 = None
        if self.phase != 'test':
            ic50 = sample['IC50 (nM)'] / 1e9
        return (torch.tensor(eval(mol_embeds)).float(), 
                torch.tensor(eval(seq_embeds)).float(), 
                torch.tensor(ic50).float())

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 dropout: float = 0.3, use_conv1: bool = False):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.conv1 = None
        if use_conv1:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor):
        left = self.process(x)
        right = self.conv1(x) if self.conv1 else x

        return F.relu(right + left)

class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 hidden_channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=2)
        )
        self.cnn = nn.Sequential(
            ResBlock(hidden_channels, out_channels, stride=1, dropout=dropout, use_conv1=True),
            ResBlock(out_channels, out_channels, stride=1, dropout=dropout),
            ResBlock(out_channels, out_channels, stride=1, dropout=dropout)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.head(x)
        x = self.cnn(x)

        return x

class DeepLPI(nn.Module):
    def __init__(self, mol_dim: int, seq_dim: int, dropout: float = 0.3):
        super().__init__()

        self.mol_dim = mol_dim
        self.seq_dim = seq_dim

        self.mol_cnn = ConvModule(1, 16)
        self.seq_cnn = ConvModule(1, 16)

        self.pool = nn.AvgPool1d(kernel_size=7, stride=5, padding=2)
        self.lstm = nn.LSTM(16, 16, num_layers=2, batch_first=True, bidirectional=True)

        self.mlp = nn.Sequential(
            nn.Linear(round(self.mol_dim / 4 + self.seq_dim / 20) * 2 * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid(),
            nn.Dropout1d(p=dropout),

            nn.Linear(1024, 1)
        )
    
    def forward(self, mol: torch.Tensor, seq: torch.Tensor):
        mol = self.mol_cnn(mol.reshape(-1, 1, self.mol_dim))
        seq = self.seq_cnn(seq.reshape(-1, 1, self.seq_dim))
        
        seq = self.pool(seq)
        x = torch.cat((mol, seq), 2)
        
        x = x.reshape(-1, round(self.mol_dim / 4 + self.seq_dim / 20), 16)
        x, _ = self.lstm(x)
        x = self.mlp(x.flatten(1))

        x = x.flatten()
        return x

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

def train_loop(model, train_dataloader, lossfunc, optimizer, device):
    model.train()
    loop_loss = 0
    total = 0
    
    progress_bar = tqdm(train_dataloader, desc="Train", postfix={"Loss": 0.000})
    
    for step, batch in enumerate(progress_bar):
        step_mol, step_seq, step_label = batch
        step_mol, step_seq, step_label = step_mol.to(device), step_seq.to(device), step_label.to(device)

        optimizer.zero_grad()
        logits = model(step_mol, step_seq)
        loss = lossfunc(logits, step_label)
        loss.backward()
        optimizer.step()
        
        loop_loss += float(loss.cpu())
        total += step_mol.shape[0]

        # Update tqdm postfix with current average loss
        progress_bar.set_postfix(Loss=loop_loss / total)
        
    
    with torch.no_grad():
        return loop_loss / len(train_dataloader)

def val_loop(model, val_loader, writer, epoch, device):
    model.eval()

    gt = []
    pred = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader)):
            step_mol, step_seq, step_label = batch
            step_mol, step_seq, step_label = step_mol.to(device), step_seq.to(device), step_label.to(device)

            logits = model(step_mol, step_seq)
            logits = logits.cpu()
            
            gt.extend(step_label.numpy().tolist())
            pred.extend(logits.numpy().tolist())

    gt = np.array(gt)
    pred = np.array(pred)
    fig = plt.figure(figsize=(6, 6))
    plt.xlabel("true value")
    plt.ylabel("predict value")
    plt.scatter(pred, gt, alpha = 0.2, color='Black')
    writer.add_figure(tag='val evaluate', figure=fig, global_step=epoch)

    return mean_squared_error(pred, gt), r2_score(pred, gt)

def main():

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = os.path.join('logs', timestamp)
    writer = SummaryWriter(log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = DTIDataset(phase="train")
    val_dataset = DTIDataset(phase="val")

    train_loader = DataLoader(train_dataset, 64, shuffle=True)
    val_loader = DataLoader(val_dataset, 64, shuffle=True)

    model = DeepLPI(mol_dim=300, seq_dim=6165, dropout=0.2).to(device)
    model.apply(initialize_weights)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, min_lr=1e-5)

    for epoch in range(1000):
        print("--"*20)
        print("epoch: " + str(epoch))
        time0 = time.time()

        avgloss = train_loop(model, train_loader, loss_fn, optimizer, device)
        msescore, r2score = val_loop(model, val_loader, writer, epoch, device)

        scheduler.step(msescore)

        writer.add_scalar("test time", time.time()-time0, epoch)
        writer.add_scalar('avgloss', avgloss , epoch)
        writer.add_scalar('mse', msescore , epoch)
        writer.add_scalar('r2', r2score , epoch)
        writer.add_scalar('current lr', optimizer.param_groups[0]['lr'], epoch)

        print()
        print("R2: " + str(r2score) + "\t MSE: " + str(msescore))
        print("use time: " + str(time.time() - time0))
        
        model.eval()
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f"model_e{epoch}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(log_dir, "quicksave.pth"))

if __name__ == "__main__":
    main()