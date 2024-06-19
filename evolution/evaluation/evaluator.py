import warnings

from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement, FieldMngt
from aquacrop.utils import prepare_weather, get_filepath
import pandas as pd
import torch
from tqdm import tqdm

from evolution.candidate import Candidate

class Evaluator:
    def __init__(self, context_length: int, weather_names: list[str]):
        """
        Valid weather names: ["tunis_climate", "brussels_climate", "hyderabad_climate", "champion_climate"]
        """
        self.maize = Crop('Maize',planting_date='05/01') # define crop
        self.loam = Soil('ClayLoam') # define soil
        self.init_wc = InitialWaterContent(wc_type='Pct',value=[70]) # define initial soil water conditions

        self.weather_names = weather_names
        self.weather_dfs, self.torch_weathers = self.load_data(self.weather_names, context_length)

    def load_data(self, weather_names: list[str], context_length: int):
        """
        Iterates over all weather files and creates a list of dataframes and torch tensors.
        The dataframes are used for the AquaCrop model and consist of each year of weather data.
        The torch tensors are used for the LSTM model to prescribe actions and consist of the preceding context_length days.
        """
        weather_dfs = []
        torch_weathers = []
        context_cols = ["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]
        for weather_name in weather_names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weather_path = get_filepath(f"{weather_name}.txt")
                weather_df = prepare_weather(weather_path)

            weather_df["tempdate"] = pd.to_datetime(weather_df["Date"])
            weather_df = weather_df.sort_values("tempdate", ascending=True)
            weather_df = weather_df.set_index("tempdate")

            # Shave off the last year because it may be incomplete
            weather_df["year"] = weather_df.index.year
            weather_df = weather_df[weather_df["year"] < weather_df["year"].max()]

            # Iterate over the unique years in the dataframe
            for year in weather_df.index.year.unique():
                # Filter the dataframe for the current year
                year_data = weather_df[weather_df.index.year == year]
                
                # Determine the start date of the preceding 90 days period
                start_date = pd.Timestamp(f'{year}-1-01') - pd.Timedelta(days=context_length)  # 90 days before January 1st of the current year
                
                # Check if there are sufficient preceding days in the dataframe
                if weather_df.index[0] > start_date:
                    continue

                # Filter the dataframe for the preceding 90 days
                preceding_data = weather_df[(weather_df.index >= start_date) & (weather_df.index < pd.Timestamp(f'{year}-01-01'))]
                
                # Append the data to the respective lists
                weather_dfs.append(year_data)
                torch_weathers.append(torch.tensor(preceding_data[context_cols].values, dtype=torch.float32, device="mps"))

        print(f"{len(weather_dfs)} data points")

        return weather_dfs, torch_weathers

    def run_model(self, weather_df, irrmngt_params, fieldmngt_params):
        """
        funciton to run model and return results for given set of soil moisture targets.
        Takes 2.5 seconds to run
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            irrmngt = IrrigationManagement(**irrmngt_params) # define irrigation management
            fieldmngt = FieldMngt(**fieldmngt_params)

            year1 = year2 = weather_df["Date"].iloc[-1].year

            # create and run model
            model = AquaCropModel(f'{year1}/05/01',
                                f'{year2}/10/31',
                                weather_df,
                                self.loam,
                                self.maize,
                                irrigation_management=irrmngt,
                                initial_water_content=self.init_wc,
                                field_management=fieldmngt)
            model.run_model(till_termination=True)
            results = model.get_simulation_results()
        return results

    def evaluate_candidate(self, candidate: Candidate):
        yields = []
        irrs = []
        mulches = []
        for weather_df, torch_weather in zip(self.weather_dfs, self.torch_weathers):
            # Run LSTM
            torch_weather = torch_weather.unsqueeze(0)
            irrmngt_params, fieldmngt_params = candidate.model.prescribe(torch_weather)
            irrmngt_params = irrmngt_params[0]
            fieldmngt_params = fieldmngt_params[0]
            # Pass actions into model
            results = self.run_model(weather_df, irrmngt_params, fieldmngt_params)
            dry_yield = results['Dry yield (tonne/ha)'].mean()
            irr = results['Seasonal irrigation (mm)'].mean()
            yields.append(dry_yield)
            irrs.append(irr)
            mulches.append(fieldmngt_params["mulch_pct"] if fieldmngt_params["mulches"] else 0)

        dry_yield = sum(yields) / len(yields)
        irr = sum(irrs) / len(irrs)
        candidate.metrics["yield"] = dry_yield
        candidate.metrics["irr"] = -1 * irr
        candidate.metrics["mulch"] = -1 * sum(mulches) / len(mulches)

        return yields, irrs

    def evaluate_candidates(self, candidates: list[Candidate]):
        for candidate in tqdm(candidates):
            if len(candidate.metrics) == 0:
                self.evaluate_candidate(candidate)
