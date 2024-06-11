from evolution.candidate import Candidate
from evolution.sorting.sorter import Sorter
from evolution.sorting.distance_calculation.distance_calculator import DistanceCalculator

class NSGA2Sorter(Sorter):
    def __init__(self, distance_calculator: DistanceCalculator):
        self.distance_calculator = distance_calculator

    def sort_candidates(self, candidates: list[Candidate]):
        # Get ranks of each candidate
        self.fast_non_dominated_sort(candidates)
        self.distance_calculator.calculate_distance(candidates)

        # Sort primarily by rank, secondarily by distance
        candidates.sort(key=lambda x: (x.rank, -x.distance))
        return candidates

    def fast_non_dominated_sort(self, candidates: list[Candidate]):
        """
        Fast non-dominated sort algorithm from ChatGPT
        """
        for c in candidates:
            c.rank = 0

        population_size = len(candidates)
        S = [[] for _ in range(population_size)]
        front = [[]]
        n = [0 for _ in range(population_size)]
        rank = [0 for _ in range(population_size)]

        for p in range(population_size):
            S[p] = []
            n[p] = 0
            for q in range(population_size):
                if self.dominates(candidates[p], candidates[q]):
                    if q not in S[p]:
                        S[p].append(q)
                elif self.dominates(candidates[q], candidates[p]):
                    n[p] = n[p] + 1
            if n[p] == 0:
                rank[p] = 0
                if p not in front[0]:
                    front[0].append(p)

        i = 0
        while front[i] != []:
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if n[q] == 0:
                        rank[q] = i+1
                        if q not in Q:
                            Q.append(q)
            i = i+1
            front.append(Q)

        # With this implementation the final front will be empty
        del front[len(front)-1]

        # Convert front indices to candidates
        candidate_fronts = []
        for f in front:
            cands = []
            for idx in f:
                candidates[idx].rank = rank[idx] + 1 # Manually increment to match NSGA-II convention
                cands.append(candidates[idx])
            candidate_fronts.append(cands)

        return candidate_fronts

    def dominates(self, candidate1: Candidate, candidate2: Candidate) -> bool:
        """
        Determine if one individual dominates another.
        """
        for obj1, obj2 in zip(candidate1.metrics, candidate2.metrics):
            if obj1 <= obj2:
                return False
        return True