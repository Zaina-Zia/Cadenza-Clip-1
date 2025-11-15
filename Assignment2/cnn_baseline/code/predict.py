import torch
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from dataset import MFCCDataset, collate_eval
from model import CNN_MFCC

def predict(data_root, split="val", batch_size=16, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    data_root = Path(data_root)

    ds = MFCCDataset(data_root, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_eval, num_workers=0)

    model = CNN_MFCC().to(device)
    model_path = data_root / "cnn_mfcc_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_ids = []

    for batch in tqdm(loader, desc=f"Predict {split}"):
        if batch is None:
            continue
        x, ids = batch
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)           # [B,1]
        all_preds.extend(pred.squeeze(1).cpu().tolist())
        all_ids.extend(ids)

    if not all_preds:
        raise RuntimeError(f"No predictions produced for split={split}. Check dataset and files.")

    df = pd.DataFrame({"signal": all_ids, "prediction": all_preds})
    out_file = data_root / f"{split}_predictions.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved predictions to {out_file}")

if __name__ == "__main__":
    data_root = r"C:\Rudaina University\Semester 3\Artificial Intelligence\Cadenza\a2\clarity_data\cad_icassp_2026\cadenza_data"
    predict(data_root, split="val")
    predict(data_root, split="eval")
