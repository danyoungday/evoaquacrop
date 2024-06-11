from pathlib import Path

import torch
from tqdm import tqdm
from aquacrop.utils import prepare_weather, get_filepath

from evolution.candidate import LSTMPrescriptor

def train_seed(seed_path: Path, max_irr_label: float):

    path = get_filepath('champion_climate.txt')
    wdf = prepare_weather(path)

    context_cols = ["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]
    torch_wdf = torch.tensor(wdf[context_cols].values, dtype=torch.float32, device="mps")

    label = torch.tensor([max_irr_label], dtype=torch.float32, device="mps")
    model = LSTMPrescriptor().to("mps")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    epochs = 250
    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            _, max_irr = model(torch_wdf)
            loss = criterion(max_irr, label)
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item()}")
            if epoch == epochs-1:
                print(max_irr.item())

    torch.save(model.state_dict(), seed_path)

if __name__ == "__main__":
    train_seed(Path("seeding/seeds/0_0.pt"), 0)
    train_seed(Path("seeding/seeds/0_1.pt"), 450)