"""
PyTorch implementation of NSGA-II.
"""
import json
from pathlib import Path

import pandas as pd
import torch

from evolution.candidate import Candidate
from evolution.crossover.uniform_crossover import UniformCrossover
from evolution.mutation.uniform_mutation import UniformMutation
from evolution.evaluation.evaluator import Evaluator
from evolution.sorting.distance_calculation.crowding_distance import CrowdingDistanceCalculator
from evolution.sorting.nsga2_sorter import NSGA2Sorter
from evolution.parent_selection.tournament_selector import TournamentSelector

class Evolution():
    def __init__(self, config: dict):

        self.save_path = Path(config["save_path"])
        self.save_path.mkdir(parents=True, exist_ok=False)
        with open(self.save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        # Evolution params
        self.evolution_params = config["evolution_params"]
        self.pop_size = self.evolution_params["pop_size"]
        self.n_generations = self.evolution_params["n_generations"]
        self.n_elites = self.evolution_params["n_elites"]
        self.seed_path = config["seed_path"]

        self.parent_selector = TournamentSelector(config["remove_population_pct"])
        self.mutator = UniformMutation(config["mutation_factor"], config["mutation_rate"])
        self.crossover = UniformCrossover(mutator=self.mutator)
        distance_calculator = CrowdingDistanceCalculator()
        self.sorter = NSGA2Sorter(distance_calculator)

        self.evaluator = Evaluator()

    def make_new_pop(self, candidates: list[Candidate], n: int, gen: int) -> list[Candidate]:
        """
        Creates new population of candidates.
        """
        children = []
        while len(children) < n:
            parents = self.parent_selector.select_parents(candidates)
            children.extend(self.crossover.crossover(f"{gen}_{len(children)}",
                                                     parents[0],
                                                     parents[1]))
        return children

    def seed_first_gen(self):
        """
        Creates the first generation by taking seeds and creating an average of them.
        """
        seed_path = Path(self.seed_path)
        candidates = []
        for seed in seed_path.iterdir():
            candidate = Candidate(seed.stem, [])
            candidate.model.load_state_dict(torch.load(seed))
            candidates.append(candidate)

        # Fill the rest of the gen with random candidates
        while len(candidates) < self.pop_size:
            candidates.extend(self.crossover.crossover(f"0_{len(candidates)}",
                                                     candidates[0],
                                                     candidates[1]))

        self.evaluator.evaluate_candidates(candidates)
        candidates = self.sorter.sort_candidates(candidates)
        self.record_gen_results(0, candidates)
        return candidates

    def record_gen_results(self, gen: int, candidates: list[Candidate]):
        """
        Logs results of generation to CSV.
        """
        gen_results = [c.record_state() for c in candidates]
        gen_results_df = pd.DataFrame(gen_results)
        csv_path = self.save_path / f"{gen}.csv"
        gen_results_df.to_csv(csv_path, index=False)

    def neuroevolution(self):
        """
        Main Neuroevolution Loop that performs NSGA-II.
        After initializing the first population randomly, goes through 3 steps in each generation:
        1. Evaluate candidates
        2. Select parents
        2a Log performance of parents
        3. Make new population from parents
        """
        print("Beginning evolution...")
        sorted_parents = self.seed_first_gen()
        offspring = []
        for gen in range(1, self.n_generations+1):
            # Create offspring
            keep = min(self.n_elites, len(sorted_parents))
            offspring = self.make_new_pop(sorted_parents, self.pop_size-keep, gen)

            # Destroy parents that weren't kept
            # if len(sorted_parents) > keep:
            #     for candidate in sorted_parents[keep:]:
            #         candidate.destroy()

            # Add elites to new generation
            # NOTE: The elites are also passed into the evaluation function. Make sure your
            # evaluator can handle this!
            new_parents = sorted_parents[:keep] + offspring
            self.evaluator.evaluate_candidates(new_parents)

            # Set rank and distance of parents
            sorted_parents = self.sorter.sort_candidates(new_parents)

            # Record the performance of the most successful candidates
            self.record_gen_results(gen, sorted_parents)

        return sorted_parents
