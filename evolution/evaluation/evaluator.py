from pathlib import Path
import warnings

from aquacrop import AquaCropModel, Soil, InitialWaterContent, Crop, IrrigationManagement, FieldMngt
from aquacrop.utils import prepare_weather, get_filepath
import pandas as pd
import torch
from tqdm import tqdm

from evolution.candidate import Candidate

class Evaluator:
    def __init__(self, tasks: list[str], soil_type: str, init_wc_params: dict, context_length: int, weather_names: list[str]):
        """
        Valid weather names: ["tunis_climate", "brussels_climate", "hyderabad_climate", "champion_climate"]
        """
        self.tasks = tasks

        self.soil = Soil(soil_type) # define soil
        self.init_wc = InitialWaterContent(**init_wc_params) # define initial soil water conditions

        self.weather_names = weather_names
        self.weather_dfs, self.torch_weathers = self.load_data(self.weather_names, context_length)

        nutrition_df = pd.read_csv(Path("data/cals.csv"))
        self.crop_cals = {row["name"]: row["kcal"] for _, row in nutrition_df.iterrows()}

    def load_data(self, weather_names: list[str], context_length: int):
        """
        Iterates over all weather files and creates a list of dataframes and torch tensors.
        The dataframes are used for the AquaCrop model and consist of each year of weather data.
        The torch tensors are used for the LSTM model to prescribe actions and consist of the preceding context_length days.
        """
        weather_dfs = []
        torch_weathers = []
        context_cols = ["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]
        normalized_cols = [f"{col}_norm" for col in context_cols]
        for weather_name in weather_names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weather_path = get_filepath(f"{weather_name}.txt")
                weather_df = prepare_weather(weather_path)

            # Get normalized context_cols for torch tensors
            for col in context_cols:
                weather_df[f"{col}_norm"] = (weather_df[col] - weather_df[col].mean()) / weather_df[col].std()

            # Sort dataframe by date
            weather_df.sort_values(by="Date", inplace=True, ascending=True)

            # Shave off the last year because it may be incomplete
            weather_df["year"] = weather_df["Date"].dt.year
            weather_df = weather_df[weather_df["year"] < weather_df["year"].max()]

            # Iterate over the unique years in the dataframe
            for year in weather_df["year"].unique():
                # We don't want the last 2 years because we need to collect 3 years' worth for the simulation
                if year == weather_df["year"].max() or year == weather_df["year"].max()-1:
                    continue

                # Filter the dataframe for the current year and the year after
                year_data = weather_df[weather_df["year"].isin([year, year + 1, year+2])]
                
                # Determine the start date of the preceding 90 days period
                start_date = pd.Timestamp(f'{year}-1-01') - pd.Timedelta(days=context_length)  # 90 days before January 1st of the current year
                
                # Check if there are sufficient preceding days in the dataframe
                if weather_df["Date"][0] > start_date:
                    continue

                # Filter the dataframe for the preceding 90 days
                preceding_data = weather_df[(weather_df["Date"] >= start_date) & (weather_df["Date"] < pd.Timestamp(f'{year}-01-01'))]
                
                # Append the data to the respective lists
                to_add = year_data.copy()
                to_add.drop(columns=normalized_cols + ["year"], inplace=True)
                weather_dfs.append(to_add)
                torch_weathers.append(torch.tensor(preceding_data[normalized_cols].values, dtype=torch.float32, device="mps"))

        print(f"Loaded {len(weather_dfs)} year pairs of weather data")
        return weather_dfs, torch_weathers

    def run_model(self, weather_df, aquacrop_params: dict):
        """
        function to run model and return results for given set of soil moisture targets.
        """
        # Convert our nice dict params into aquacrop objects
        model_params = {}
        model_params["crop"] = Crop(**aquacrop_params["crop"])
        if "irrigation_management" in aquacrop_params:
            model_params["irrigation_management"] = IrrigationManagement(**aquacrop_params["irrigation_management"])
        if "field_management" in aquacrop_params:
            model_params["field_management"] = FieldMngt(**aquacrop_params["field_management"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            year1 = weather_df["Date"].min().year
            start_date = f"{year1}/{aquacrop_params['crop']['planting_date']}"
            year2 = weather_df["Date"].max().year
            # create and run model
            model = AquaCropModel(start_date,
                                f'{year2}/12/31',
                                weather_df,
                                self.soil,
                                initial_water_content=self.init_wc,
                                **model_params)
            model.run_model(till_termination=True)
            results = model.get_simulation_results()
        return results

    def evaluate_candidate(self, candidate: Candidate):
        yields = []
        irrs = []
        for weather_df, torch_weather in zip(self.weather_dfs, self.torch_weathers):
            # Run LSTM
            torch_weather = torch_weather.unsqueeze(0)
            aquacrop_params = candidate.prescribe(torch_weather)
            aquacrop_params = aquacrop_params[0]
            # Pass actions into model
            results = self.run_model(weather_df, aquacrop_params)
            dry_yield = results['Dry yield (tonne/ha)'].iloc[0]
            if "kcal" in self.tasks:
                crop_name = aquacrop_params["crop"]["c_name"]
                kcal = self.crop_cals[crop_name]
                dry_yield *= kcal * 10000
            irr = results['Seasonal irrigation (mm)'].iloc[0]
            yields.append(dry_yield)
            irrs.append(irr)

        dry_yield = sum(yields) / len(yields)
        irr = sum(irrs) / len(irrs)
        if "kcal" in self.tasks:
            candidate.metrics["kcal"] = dry_yield
        else:
            candidate.metrics["yield"] = dry_yield
        candidate.metrics["irr"] = -1 * irr

        return yields, irrs

    def evaluate_candidates(self, candidates: list[Candidate]):
        for candidate in tqdm(candidates):
            if len(candidate.metrics) == 0:
                self.evaluate_candidate(candidate)
