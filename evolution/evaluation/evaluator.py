import warnings

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import torch
from tqdm import tqdm

class Evaluator:
    def __init__(self):
        path = get_filepath('champion_climate.txt')
        self.wdf = prepare_weather(path)
        self.maize = Crop('Maize',planting_date='05/01') # define crop
        self.loam = Soil('ClayLoam') # define soil
        self.init_wc = InitialWaterContent(wc_type='Pct',value=[70]) # define initial soil water conditions

        context_cols = ["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]
        self.torch_wdf = torch.tensor(self.wdf[context_cols].values, dtype=torch.float32, device="mps")

    def run_model(self, smts, max_irr_season, year1, year2):
        """
        funciton to run model and return results for given set of soil moisture targets.
        Takes 2.5 seconds to run
        """
        irrmngt = IrrigationManagement(irrigation_method=1,SMT=smts,MaxIrrSeason=max_irr_season) # define irrigation management

        # create and run model
        model = AquaCropModel(f'{year1}/05/01', 
                            f'{year2}/10/31',
                            self.wdf,
                            self.loam,
                            self.maize,
                            irrigation_management=irrmngt,
                            initial_water_content=self.init_wc)

        model.run_model(till_termination=True)
        return model.get_simulation_results()

    def evaluate_candidate(self, candidate):
        smts, max_irr_season = candidate.model(self.torch_wdf)
        smts = smts.detach().cpu().numpy()
        max_irr_season = max_irr_season.item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self.run_model(smts, max_irr_season, 2018, 2018)
        dry_yield = results['Dry yield (tonne/ha)'].mean()
        irr = results['Seasonal irrigation (mm)'].mean()

        candidate.metrics = [dry_yield, -1 * irr]
        candidate.smts = list(smts)
        candidate.max_irr_season = max_irr_season

    def evaluate_candidates(self, candidates):
        for candidate in tqdm(candidates):
            if candidate.metrics is None:
                self.evaluate_candidate(candidate)
    