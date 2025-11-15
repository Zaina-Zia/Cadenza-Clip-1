import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from dataset import MFCCDataset, collate_train
from model import CNN_MFCC

def train(data_root,
          batch_size=32,
          epochs=10,
          lr=1e-3,
          device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    data_root = Path(data_root)

    train_ds = MFCCDataset(data_root, split="train")
    val_ds   = MFCCDataset(data_root, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_train, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_train, num_workers=0)

    model = CNN_MFCC().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        skipped_train = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            if batch is None:
                skipped_train += 1
                continue
            x, y, _ = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)           # [B,1]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = sum(train_losses) / len(train_losses) if train_losses else float("nan")

        # validation
        model.eval()
        val_losses = []
        skipped_val = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                if batch is None:
                    skipped_val += 1
                    continue
                x, y, _ = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())

        avg_val = sum(val_losses) / len(val_losses) if val_losses else float("nan")

        print(f"Epoch {epoch}: Train Loss={avg_train:.4f} (skipped {skipped_train}), Val Loss={avg_val:.4f} (skipped {skipped_val})")

    # save
    out = data_root / "cnn_mfcc_model.pt"
    torch.save(model.state_dict(), out)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    data_root = r"C:\Rudaina University\Semester 3\Artificial Intelligence\Cadenza\a2\clarity_data\cad_icassp_2026\cadenza_data"
    train(data_root, batch_size=32, epochs=10, lr=1e-3, device="cuda")
