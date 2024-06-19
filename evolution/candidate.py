from pathlib import Path

import torch

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
            sigmoid * 450
            mm from 0-450
        [mulchpct]:
            tanh * 100
            % from 0-100. Mulches is set to False if <= 0
        [fmulch]:
            sigmoid
            factor from 0-1
        [zBund]:
            tanh
            m from 0-1. Bunds is set to False if <= 0
        [BundWater]:
            sigmoid * 450
            mm from 0-450
        [CNadjPct]:
            tanh * 100
            % from -100-100
        [SRinhb]:
            sigmoid
            bool False if <= 0.5
        """
        # TODO: Should we do anything about these?
        h0 = torch.randn((1, x.shape[0], self.hidden_size)).to("mps")
        c0 = torch.randn((1, x.shape[0], self.hidden_size)).to("mps")
        _, (hn, _) = self.lstm(x, (h0, c0))
        outputs = self.linear(hn)
        outputs = outputs.squeeze(1)

        smts = torch.sigmoid(outputs[:, :4]) * 100
        max_irr_season = torch.sigmoid(outputs[:, 4]).unsqueeze(1) * 450
        mulchpct = torch.tanh(outputs[:, 5]).unsqueeze(1) * 100
        fmulch = torch.sigmoid(outputs[:, 6]).unsqueeze(1)
        zbund = torch.tanh(outputs[:, 7]).unsqueeze(1)
        bundwater = torch.sigmoid(outputs[:, 8]).unsqueeze(1) * 450
        cnadjpct = torch.tanh(outputs[:, 9]).unsqueeze(1) * 100
        srinhb = torch.sigmoid(outputs[:, 10]).unsqueeze(1)
        combined = torch.cat([smts, max_irr_season, mulchpct, fmulch, zbund, bundwater, cnadjpct, srinhb], dim=1)
        return combined
    
    def prescribe(self, x):
        """
        Parses the output of our model so that we can use it in the AquaCrop model.
        """
        outputs = self.forward(x).detach().cpu().numpy()
        irrmngts = []
        fieldmngts = []
        for batch in outputs:
            irrmngt_params = {"irrigation_method": 1}
            irrmngt_params["SMT"] = batch[:4]
            # irrmngt_params["max_irr_season"] = batch[4]

            fieldmngt_params = {}
            fieldmngt_params["mulches"] = batch[5] > 0
            if fieldmngt_params["mulches"]:
                fieldmngt_params["mulch_pct"] = batch[5]
            
            # TODO: Re-add these later. Looks like fmulch or bunds is breaking it
            # fieldmngt_params["f_mulch"] = batch[6]
            # fieldmngt_params["bunds"] = batch[7] > 0
            # if fieldmngt_params["bunds"]:
            #     fieldmngt_params["z_bund"] = batch[7]
            # fieldmngt_params["bund_water"] = batch[8]
            # fieldmngt_params["curve_number_adj"] = batch[9] != 0 # TODO: This will never be false!
            # fieldmngt_params["curve_number_adj_pct"] = batch[9]
            # fieldmngt_params["sr_inhb"] = batch[10] > 0.5

            irrmngts.append(irrmngt_params)
            fieldmngts.append(fieldmngt_params)
        return irrmngts, fieldmngts


