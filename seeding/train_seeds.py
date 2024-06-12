from pathlib import Path
import warnings

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from aquacrop.utils import prepare_weather, get_filepath

from evolution.candidate import LSTMPrescriptor

def load_data(weather_names: list[str]):
        # TODO This is copy-pasted from evaluator.py
        weather_dfs = []
        for weather_name in weather_names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weather_path = get_filepath(f"{weather_name}.txt")
                weather_df = prepare_weather(weather_path)
            weather_dfs.append(weather_df)

        context_cols = ["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]
        context_values = [torch.tensor(df[context_cols].values, dtype=torch.float32) for df in weather_dfs]

        min_length = min(value.shape[0] for value in context_values)
        truncated = [value[:min_length] for value in context_values]
        torch_weather = torch.stack(truncated).to("mps")

        return weather_dfs, torch_weather

def train_seed(seed_path: Path, label: list[float]):

    _, torch_weather = load_data(["tunis_climate", "brussels_climate", "hyderabad_climate", "champion_climate"])
    ds = TensorDataset(torch_weather)
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
            for torch_wdf in dl:
                optimizer.zero_grad()
                smts, max_irr = model(torch_wdf[0])
                output = torch.cat([smts, max_irr.unsqueeze(0)], dim=1)
                loss = criterion(output, label_tensor)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            pbar.set_description(f"Avg Loss: {avg_loss / len(ds)}")

    torch.save(model.state_dict(), seed_path)

if __name__ == "__main__":
    seed_dir = Path("seeding/seeds/fourpoint")
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_seed(seed_dir / "0_0.pt", [0, 0, 0, 0, 0])
    train_seed(seed_dir / "0_1.pt", [100, 100, 100, 100, 450])