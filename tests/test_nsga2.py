import unittest

from evolution.sorting.distance_calculation.crowding_distance import CrowdingDistanceCalculator
from evolution.sorting.nsga2_sorter import NSGA2Sorter

class TestNSGA2(unittest.TestCase):

    class Candidate():
        def __init__(self, metrics: dict):
            self.metrics = metrics
            self.rank = None
            self.distance = None

        def __repr__(self):
            return f"Candidate({self.metrics})"

    def setUp(self):
        self.sorter = NSGA2Sorter(CrowdingDistanceCalculator())
    
    def test_non_dominated_sort(self):
        candidates = [
            self.Candidate({"a": 1, "b": 2}),
            self.Candidate({"a": 0, "b": 0}),
            self.Candidate({"a": 2, "b": 1}),
            self.Candidate({"a": 1, "b": 1}),
            self.Candidate({"a": 3, "b": 3})
        ]
        fronts = self.sorter.fast_non_dominated_sort(candidates)
        ranks = [2, 3, 2, 2, 1]
        for rank, front in enumerate(fronts):
            for candidate in front:
                self.assertEqual(candidate.rank, rank+1)
        for rank, candidate in zip(ranks, candidates):
            self.assertEqual(candidate.rank, rank)

    def test_weird_zero_case(self):
        candidates = [
            self.Candidate({"a": 0.0, "b": -0.0}),
            self.Candidate({"a": 0.0, "b": -0.0}),
            self.Candidate({"a": 14.25381137854931,"b": -604.25}),
            self.Candidate({"a": 0.0, "b": -10})
        ]
        fronts = self.sorter.fast_non_dominated_sort(candidates)
        ranks = [1, 1, 1, 2]

        for rank, front in enumerate(fronts):
            for candidate in front:
                self.assertEqual(candidate.rank, rank+1)
        for rank, candidate in zip(ranks, candidates):
            self.assertEqual(candidate.rank, rank)


