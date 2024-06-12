import warnings

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import torch
from tqdm import tqdm

from evolution.candidate import Candidate

class Evaluator:
    def __init__(self):
        self.maize = Crop('Maize',planting_date='05/01') # define crop
        self.loam = Soil('ClayLoam') # define soil
        self.init_wc = InitialWaterContent(wc_type='Pct',value=[70]) # define initial soil water conditions

        weather_names = ["tunis_climate", "brussels_climate", "hyderabad_climate", "champion_climate"]
        self.weather_dfs, self.torch_weather = self.load_data(weather_names)

        self.year1 = self.year2 = 2000

    def load_data(self, weather_names: list[str]):
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

    def run_model(self, weather_df, smts, max_irr_season):
        """
        funciton to run model and return results for given set of soil moisture targets.
        Takes 2.5 seconds to run
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            irrmngt = IrrigationManagement(irrigation_method=1,SMT=smts,MaxIrrSeason=max_irr_season) # define irrigation management

            # create and run model
            model = AquaCropModel(f'{self.year1}/05/01', 
                                f'{self.year2}/10/31',
                                weather_df,
                                self.loam,
                                self.maize,
                                irrigation_management=irrmngt,
                                initial_water_content=self.init_wc)
            model.run_model(till_termination=True)
            results = model.get_simulation_results()
        return results

    def evaluate_candidate(self, candidate: Candidate):
        batch_smts, batch_max_irr_season = candidate.model(self.torch_weather)
        batch_smts = batch_smts.detach().cpu().numpy()
        yields = []
        irrs = []
        for i, weather_df in enumerate(self.weather_dfs):
            results = self.run_model(weather_df, batch_smts[i], batch_max_irr_season[i].item())
            dry_yield = results['Dry yield (tonne/ha)'].mean()
            irr = results['Seasonal irrigation (mm)'].mean()
            yields.append(dry_yield)
            irrs.append(irr)

        dry_yield = sum(yields) / len(yields)
        irr = sum(irrs) / len(irrs)
        candidate.metrics = [dry_yield, -1 * irr]

    def evaluate_candidates(self, candidates: list[Candidate]):
        for candidate in tqdm(candidates):
            if candidate.metrics is None:
                self.evaluate_candidate(candidate)
