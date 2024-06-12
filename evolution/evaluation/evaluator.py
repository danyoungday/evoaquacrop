import warnings

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
from aquacrop.utils import prepare_weather, get_filepath
import pandas as pd
import torch
from tqdm import tqdm

from evolution.candidate import Candidate

class Evaluator:
    def __init__(self, weather_names=["tunis_climate", "brussels_climate", "hyderabad_climate", "champion_climate"]):
        self.maize = Crop('Maize',planting_date='05/01') # define crop
        self.loam = Soil('ClayLoam') # define soil
        self.init_wc = InitialWaterContent(wc_type='Pct',value=[70]) # define initial soil water conditions

        self.weather_names = weather_names
        self.weather_dfs, self.torch_weathers = self.load_data(self.weather_names)

    def load_data(self, weather_names: list[str]):
        weather_dfs = []
        torch_weathers = []
        context_cols = ["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]
        for weather_name in weather_names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weather_path = get_filepath(f"{weather_name}.txt")
                weather_df = prepare_weather(weather_path)

            weather_df["tempdate"] = pd.to_datetime(weather_df["Date"])
            weather_df["year"] = weather_df["tempdate"].dt.year
            # Shave off the last year because it may be incomplete
            df_subset = weather_df[weather_df["year"] < weather_df["year"].max()]

            # Torch should get one year less than this because it can't see the future
            torch_subset = df_subset[df_subset["year"] < df_subset["year"].max()]
            torch_weathers.append(torch.tensor(torch_subset[context_cols].values, dtype=torch.float32, device="mps"))

            weather_dfs.append(df_subset.drop(columns=["year", "tempdate"]))

        torch_weathers = [torch.tensor(df[context_cols].values, dtype=torch.float32, device="mps") for df in weather_dfs]

        return weather_dfs, torch_weathers

    def run_model(self, weather_df, smts, max_irr_season):
        """
        funciton to run model and return results for given set of soil moisture targets.
        Takes 2.5 seconds to run
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            irrmngt = IrrigationManagement(irrigation_method=1,SMT=smts,MaxIrrSeason=max_irr_season) # define irrigation management

            year1 = year2 = weather_df["Date"].iloc[-1].year

            # create and run model
            model = AquaCropModel(f'{year1}/05/01',
                                f'{year2}/10/31',
                                weather_df,
                                self.loam,
                                self.maize,
                                irrigation_management=irrmngt,
                                initial_water_content=self.init_wc)
            model.run_model(till_termination=True)
            results = model.get_simulation_results()
        return results

    def evaluate_candidate(self, candidate: Candidate):
        yields = []
        irrs = []
        for weather_df, torch_weather in zip(self.weather_dfs, self.torch_weathers):
            # Run LSTM
            torch_weather = torch_weather.unsqueeze(0)
            smts, max_irr_season = candidate.model(torch_weather)
            smts = smts.squeeze().detach().cpu().numpy()
            max_irr_season = max_irr_season.item()

            # Pass actions into model
            results = self.run_model(weather_df, smts, max_irr_season)
            dry_yield = results['Dry yield (tonne/ha)'].mean()
            irr = results['Seasonal irrigation (mm)'].mean()
            yields.append(dry_yield)
            irrs.append(irr)

        dry_yield = sum(yields) / len(yields)
        irr = sum(irrs) / len(irrs)
        candidate.metrics = [dry_yield, -1 * irr]

        candidate.full_yields = yields
        candidate.full_irrs = irrs

        return yields, irrs

    def evaluate_candidates(self, candidates: list[Candidate]):
        for candidate in tqdm(candidates):
            if candidate.metrics is None:
                self.evaluate_candidate(candidate)

def main():
    evaluator = Evaluator()
    candidate = Candidate("0_0", [])
    evaluator.evaluate_candidate(candidate)
    print(candidate.metrics)

if __name__ == "__main__":
    main()