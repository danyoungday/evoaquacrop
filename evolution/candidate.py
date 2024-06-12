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
        self.model = LSTMPrescriptor(4, 16, 5).to("mps")
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
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Returns shapes [4], [1]
        """
        # TODO: Should we do anything about these?
        h0 = torch.randn((1, x.shape[0], self.hidden_size)).to("mps")
        c0 = torch.randn((1, x.shape[0], self.hidden_size)).to("mps")
        _, (hn, _) = self.lstm(x, (h0, c0))
        outputs = self.linear(hn)
        outputs = torch.sigmoid(outputs)
        smts = outputs[0, :, :4] * 100
        max_irr_season = outputs[0, :, 4] * 450

        return smts, max_irr_season

