import torch

class Candidate():
    """
    Candidate class that points to a model on disk.
    """
    def __init__(self, cand_id: str, parents: list[str]):
        self.cand_id = cand_id
        self.metrics = None
        self.smts = None
        self.max_irr_season = None

        self.parents = parents
        self.rank = None
        self.distance = None

        # Model
        # TODO: Some sort of different initialization
        self.model = LSTMPrescriptor().to("mps")
        self.model.eval()

    def record_state(self):
        """
        Records metrics as well as seed and parents for reconstruction.
        """
        state = {
            "cand_id": self.cand_id,
            "parents": self.parents,
            "rank": self.rank,
            "distance": self.distance,
            "yield": self.metrics[0],
            "irrigation": self.metrics[1],
            "smts": self.smts,
            "max_irr_season": self.max_irr_season
        }
        return state

    def __str__(self):
        return f"Candidate({self.cand_id})"
    
    def __repr__(self):
        return f"Candidate({self.cand_id})"

class LSTMPrescriptor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(4, 16, batch_first=True)
        self.linear = torch.nn.Linear(16, 5)

    def forward(self, x):
        """
        Returns shapes [4], [1]
        """
        # TODO: Should we do anything about these?
        h0 = torch.randn((1, 16)).to("mps")
        c0 = torch.randn((1, 16)).to("mps")
        _, (hn, _) = self.lstm(x, (h0, c0))
        outputs = self.linear(hn)
        smts = torch.sigmoid(outputs[:, :4]) * 100
        max_irr_season = torch.sigmoid(outputs[:, 4]) * 450 # TODO: Abs vs. relu?

        return smts.squeeze(), max_irr_season

