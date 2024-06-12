from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from evolution.candidate import LSTMPrescriptor
from evolution.evaluation.evaluator import Evaluator

class CustomDS(Dataset):
    def __init__(self, torch_weathers: list[torch.Tensor]):
        self.x = torch_weathers

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

def train_seed(seed_path: Path, label: list[float]):

    evaluator = Evaluator()
    _, torch_weathers = evaluator.load_data(["tunis_climate", "brussels_climate", "hyderabad_climate", "champion_climate"])
    ds = CustomDS(torch_weathers)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    label_tensor = torch.tensor([label], dtype=torch.float32, device="mps")
    model = LSTMPrescriptor(4, 16, 5)
    model.to("mps")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    epochs = 250
    with tqdm(range(epochs)) as pbar:
        for _ in pbar:
            avg_loss = 0
            for torch_weather in dl:
                optimizer.zero_grad()
                smts, max_irr = model(torch_weather)
                output = torch.cat([smts, max_irr.unsqueeze(0)], dim=1)
                loss = criterion(output, label_tensor)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            pbar.set_description(f"Avg Loss: {avg_loss / len(ds)}")

    torch.save(model.state_dict(), seed_path)

if __name__ == "__main__":
    seed_dir = Path("seeding/seeds/fourpointtwo")
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_seed(seed_dir / "0_0.pt", [0, 0, 0, 0, 0])
    train_seed(seed_dir / "0_1.pt", [100, 100, 100, 100, 450])