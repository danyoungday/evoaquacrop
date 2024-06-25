from datetime import datetime, timedelta
from pathlib import Path

import torch

from evolution.constants import CROP_NAMES

class Candidate():
    """
    Candidate class that points to a model on disk.
    """
    def __init__(self, cand_id: str, parents: list[str], model_params: dict, tasks: list[str]):
        self.cand_id = cand_id
        self.tasks = tasks
        self.metrics = {}
        self.full_yields = None
        self.full_irrs = None

        self.parents = parents
        self.rank = None
        self.distance = None

        # Model
        self.model_params = model_params
        self.model = LSTMPrescriptor(**model_params).to("mps")
        self.model.eval()

    @classmethod
    def from_seed(cls, path: Path, model_params: dict, tasks: list[str]):
        cand_id = path.stem
        parents = []
        candidate = cls(cand_id, parents, model_params, tasks)
        candidate.model.load_state_dict(torch.load(path))
        return candidate
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def prescribe(self, x: torch.tensor) -> dict:
        """
        Parses the output of our model so that we can use it in the AquaCrop model.
        """
        outputs = self.model.forward(x).detach().cpu().numpy()
        aquacrop_params = []
        for batch in outputs:
            i = 0
            irrmngt_params = {"irrigation_method": 1}
            irrmngt_params["SMT"] = batch[i:i+4]
            i += 4
            irrmngt_params["max_irr_season"] = batch[i]
            i += 1

            # fieldmngt_params = {}
            # fieldmngt_params["mulches"] = batch[i] > 0
            # if fieldmngt_params["mulches"]:
            #     fieldmngt_params["mulch_pct"] = batch[i]
            # i += 1
            
            # fieldmngt_params["bunds"] = batch[i] > 0
            # if fieldmngt_params["bunds"]:
            #     fieldmngt_params["z_bund"] = batch[i] + 0.001 # zbund must be greater than 0.001
            #     fieldmngt_params["bund_water"] = batch[i+1]
            # i += 2

            crop = {}
            crop_probs = batch[i:i+len(CROP_NAMES)]
            crop_idx = crop_probs.argmax()
            crop["c_name"] = CROP_NAMES[crop_idx]
            i += len(CROP_NAMES)
            planting_date = datetime(2001, 1, 1) + timedelta(days=int(batch[i]))
            planting_date = planting_date.strftime("%m/%d")
            crop["planting_date"] = planting_date
            i += 1

            params = {}
            params["irrigation_management"] = irrmngt_params
            # params["field_management"] = fieldmngt_params
            params["crop"] = crop
            aquacrop_params.append(params)
        return aquacrop_params

    def record_state(self):
        """
        Records metrics as well as seed and parents for reconstruction.
        """
        state = {
            "cand_id": self.cand_id,
            "parents": self.parents,
            "rank": self.rank,
            "distance": self.distance,
        }
        for task in self.tasks:
            state[task] = self.metrics[task]
        return state

    def __str__(self):
        return f"Candidate({self.cand_id})"
    
    def __repr__(self):
        return f"Candidate({self.cand_id})"

class LSTMPrescriptor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Outputs:
        [smt0, smt1, smt2, smt3]:
            sigmoid * 100
            % from 0-100
        [max_irr_season]:
            abs
            Unlimited from 0-inf
        [mulchpct]:
            sigmoid * 100
            % from 0-100. Mulches is set to False if 0
        [zBund]:
            abs
            m from 0-inf. Bunds is set to False if 0
        [BundWater]:
            abs
            mm from 0-inf
        [crop]:
            17 crop types (which doesn't include GDD or Default) softmaxed
        [planting_date]:
            date from jan 1 - dec 31
        """
        # TODO: Should we do anything about these?
        h0 = torch.randn((1, x.shape[0], self.hidden_size)).to("mps")
        c0 = torch.randn((1, x.shape[0], self.hidden_size)).to("mps")
        _, (hn, _) = self.lstm(x, (h0, c0))
        outputs = self.linear(hn)
        outputs = outputs.squeeze(1)

        processed = []
        i = 0
        smts = torch.sigmoid(outputs[:, i:i+4]) * 100
        processed.append(smts)
        i += 4
        max_irr_season = torch.sigmoid(outputs[:, i]).unsqueeze(1) * 1000
        processed.append(max_irr_season)
        i += 1
        # mulchpct = torch.sigmoid(outputs[:, i]).unsqueeze(1) * 100
        # i += 1
        # processed.append(mulchpct)
        # zbund = torch.abs(outputs[:, i]).unsqueeze(1)
        # i += 1
        # processed.append(zbund)
        # bundwater = torch.abs(outputs[:, i]).unsqueeze(1)
        # i += 1
        # processed.append(bundwater)
        crop_logits = outputs[:, i:i+len(CROP_NAMES)]
        i += len(CROP_NAMES)
        crop_probs = torch.softmax(crop_logits, dim=1)
        processed.append(crop_probs)
        planting_date = torch.sigmoid(outputs[:, i]).unsqueeze(1) * 364
        processed.append(planting_date)
        i += 1
        combined = torch.cat(processed, dim=1)
        assert combined.shape[1] == outputs.shape[1], f"{combined.shape[1]} != {outputs.shape[1]}"
        return combined
