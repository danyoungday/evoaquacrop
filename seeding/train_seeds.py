import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from evolution.constants import CROP_NAMES
from evolution.candidate import LSTMPrescriptor
from evolution.evaluation.evaluator import Evaluator

class CustomDS(Dataset):
    def __init__(self, torch_weathers: list[torch.Tensor]):
        self.x = torch_weathers

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

def train_seed(epochs: int, model_params: dict, seed_path: Path, torch_weathers: list[torch.tensor], label: torch.tensor):
    ds = CustomDS(torch_weathers)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    label_tensor = label.to("mps")
    model = LSTMPrescriptor(**model_params)
    model.to("mps")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()
    avg_loss = 0
    with tqdm(range(epochs), leave=False) as pbar:
        for _ in pbar:
            avg_loss = 0
            for torch_weather in dl:
                optimizer.zero_grad()
                output = model(torch_weather).squeeze()
                loss = criterion(output, label_tensor)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= len(ds)
            pbar.set_postfix({"Loss": avg_loss})

    torch.save(model.state_dict(), seed_path)
    return avg_loss

def create_labels():
    """
    WARNING: Labels have to be added in the exact same order as the model.
    """
    categories = []
    irr_max = torch.tensor([100, 100, 100, 100, 1000], dtype=torch.float32)
    irr_min = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
    irrs = [irr_min, irr_max]
    categories.append(irrs)

    # mulch_max = torch.tensor([100], dtype=torch.float32)
    # mulch_min = torch.tensor([0], dtype=torch.float32)
    # mulches = [mulch_min, mulch_max]
    # categories.append(mulches)

    # bund_max = torch.tensor([2, 1000], dtype=torch.float32)
    # bund_min = torch.tensor([0, 0], dtype=torch.float32)
    # bunds = [bund_min, bund_max]
    # categories.append(bunds)

    crops = []
    for i in range(len(CROP_NAMES)):
        crops.append(torch.tensor([0 if i != j else 1 for j in range(len(CROP_NAMES))], dtype=torch.float32))
    categories.append(crops)

    jan1 = datetime(2001, 1, 1)
    apr1 = datetime(2001, 4, 1)
    oct1 = datetime(2001, 10, 1)
    planting_date_max = torch.tensor([(apr1-jan1).days], dtype=torch.float32)
    planting_date_min = torch.tensor([(oct1-jan1).days], dtype=torch.float32)
    planting_dates =[planting_date_min, planting_date_max]
    categories.append(planting_dates)

    labels = []
    combinations = list(itertools.product(*categories))
    for combination in combinations:
        labels.append(torch.cat(combination, dim=0))

    return labels

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument("--epochs", type=int, default=250, help="Number of epochs to train for.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(config)

    if Path(config["seed_path"]).exists():
        inp = input("Seed path already exists, do you want to overwrite? (y/n):")
        if inp.lower() == "y":
            shutil.rmtree(config["seed_path"])
        else:
            print("Exiting")
            exit()

    evaluator_params = config["evaluation_params"]
    evaluator = Evaluator(**evaluator_params)
    torch_weathers = evaluator.torch_weathers
    model_params = config["model_params"]
    seed_dir = Path(config["seed_path"])
    seed_dir.mkdir(parents=True, exist_ok=True)

    labels = create_labels()
    torch.manual_seed(42)
    with tqdm(enumerate(labels), total=len(labels)) as pbar:
        for i, label in pbar:
            pbar.set_description(f"Training seed 0_{i}")
            final_loss = train_seed(args.epochs, model_params, seed_dir / f"0_{i}.pt", torch_weathers, label)
            pbar.set_postfix({"Loss": final_loss})

if __name__ == "__main__":
    main()
