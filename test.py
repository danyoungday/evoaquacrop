import time

from evolution.candidate import Candidate
from evolution.mutation.uniform_mutation import UniformMutation
from evolution.crossover.uniform_crossover import UniformCrossover
from evolution.evaluation.evaluator import Evaluator

def main():
    s = time.time()
    candidate = Candidate("0_0", ["0_0, 0_0"])
    evaluator = Evaluator()
    evaluator.evaluate_candidate(candidate)
    e = time.time()
    print(e - s)
    print(candidate.metrics)

if __name__ == "__main__":
    main()