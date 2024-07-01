from pathlib import Path
import random
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
        
        # self.soil = Soil(soil_type) # define soil
        # self.init_wc = InitialWaterContent(**init_wc_params) # define initial soil water conditions

        self.weather_names = weather_names
        self.context_cols = ["MinTemp", "MaxTemp", "Precipitation", "ReferenceET"]
        self.normalized_cols = [f"{col}_norm" for col in self.context_cols]
        self.weather_dfs, self.torch_weathers = self.load_data(self.weather_names, context_length)

        # Artificially create soils
        random.seed(42)
        soil_types = [
            'Clay', 'ClayLoam', 'Loam', 'LoamySand', 'Sand', 'SandyClay', 
            'SandyClayLoam', 'SandyLoam', 'Silt', 'SiltClayLoam', 
            'SiltLoam', 'SiltClay', 'Paddy'
        ]
        soils = random.choices(soil_types, k=len(self.weather_dfs))
        self.soils = [Soil(soil) for soil in soils]
        wcs = [random.random() * 10 for _ in range(len(self.weather_dfs))]
        self.init_wcs = [InitialWaterContent(wc_type="Pct", value=[wc]) for wc in wcs]

        nutrition_df = pd.read_csv(Path("data/cals.csv"))
        self.crop_cals = {row["name"]: row["kcal"] for _, row in nutrition_df.iterrows()}

    def load_data(self, weather_names: list[str], context_length: int):
        """
        Loads weather data.
        Scales the context columns across all countries.
        Then goes through each country and gets sliding windows of 3 years of weather data for the simulator
        + the preceding context_length days of weather data for the LSTM.
        """
        country_dfs = []
        # Load each country's weather df
        for weather_name in weather_names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weather_path = get_filepath(f"{weather_name}.txt")
                country_df = prepare_weather(weather_path)
                country_dfs.append(country_df)
        
        # Get scaled context column across all countries
        combined_df = pd.concat(country_dfs)
        assert len(combined_df) == sum([len(df) for df in country_dfs])
        for country_df in country_dfs:
            for col in self.context_cols:
                country_df[f"{col}_norm"] = (country_df[col] - combined_df[col].mean()) / combined_df[col].std()

        # Assemble inputs
        weather_dfs = []
        torch_weathers = []
        for country_df in country_dfs:
            s = len(weather_dfs)
            country_df = country_df.sort_values("Date")
            # Shave off the last year because it may be incomplete
            country_df["year"] = country_df["Date"].dt.year
            country_df = country_df[country_df["year"] < country_df["year"].max()]

            # Iterate over the unique years in the dataframe
            for year in country_df["year"].unique():
                # We don't want the last 2 years because we need to collect 3 years' worth for the simulation
                if year == country_df["year"].max() or year == country_df["year"].max()-1:
                    continue
                
                # Determine the start date of the preceding 90 days period
                torch_start_date = pd.Timestamp(f'{year}-1-01') - pd.Timedelta(days=context_length)  # 90 days before January 1st of the current year
                # We don't want to include the first year if it doesn't have enough preceding days
                if country_df["Date"][0] > torch_start_date:
                    continue

                # Filter the dataframe for the current year and the year after
                year_data = country_df[country_df["year"].isin([year, year+1, year+2])]

                # Append the data to the respective lists
                to_add = year_data.copy()
                to_add.drop(columns=["year"] + self.normalized_cols, inplace=True)
                weather_dfs.append(to_add)

                # Construct torch tensor for LSTM from preceding 90 days
                preceding_data = country_df[(country_df["Date"] >= torch_start_date) & (country_df["Date"] < pd.Timestamp(f'{year}-01-01'))]
                assert len(preceding_data) == context_length, f"Expected {context_length} days for year {year}, got {len(preceding_data)}"
                torch_weathers.append(torch.tensor(preceding_data[self.normalized_cols].values, dtype=torch.float32, device="mps"))

            print(f"Loaded {len(weather_dfs) - s} year pairs of weather data")
        
        return weather_dfs, torch_weathers

    def run_model(self, weather_df, soil, init_wc, aquacrop_params: dict):
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
                                soil=soil,
                                initial_water_content=init_wc,
                                **model_params)
            model.run_model(till_termination=True)
            results = model.get_simulation_results()
        return results

    def evaluate_candidate(self, candidate: Candidate):
        yields = []
        irrs = []
        for weather_df, soil, init_wc, torch_weather in zip(self.weather_dfs, self.soils, self.init_wcs, self.torch_weathers):
            # Run LSTM
            torch_weather = torch_weather.unsqueeze(0)
            aquacrop_params = candidate.prescribe(torch_weather)
            aquacrop_params = aquacrop_params[0]
            # Pass actions into model
            results = self.run_model(weather_df, soil, init_wc, aquacrop_params)
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
