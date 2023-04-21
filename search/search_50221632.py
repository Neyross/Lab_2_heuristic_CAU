from problem import RandomGeometricGraphProblem
from pathlib import Path
from typing import List
from queue import PriorityQueue


STUDENT_ID = Path(__file__).stem.split('_')[1]


class Assignment:
    def __init__(self, problem: RandomGeometricGraphProblem):
        self.problem = problem

    def heuristic(self, state: str) -> float:
        goal = self.problem.places_to_visit[-1]
        s1, s2 = self.problem.get_position_of(state)
        g1, g2 = self.problem.get_position_of(goal)

        return abs(s1 - g1) + abs(s2 - g2)

    def search(self, criteria: tuple, time_limit: int) -> List[str]:
        """
        IDA* algorithm
        - I selected IDA* because it's an informed search algorithm that can be memory-efficient by exploring only the most promising
        nodes, similar to A*, but it doesn't need to store all the nodes in the memory like A*.
        - frontier is initialized with the heuristic value of the initial state
        - reached is a set of tuples containing visited nodes
        - threshold is the maximum cost allowed for the path
        """
        threshold = float('inf')
        frontier = PriorityQueue()
        frontier.put((self.heuristic(self.problem.initial_state), (self.problem.initial_state,), 0))

        reached = set({(self.problem.initial_state,)})

        while not frontier.empty():
            _, path, cost = frontier.get()
            if self.problem.is_solution_path(path):
                return path

            for child, child_criteria in self.problem.expand(path[-1]):
                if child in reached:
                    continue

                child_cost = sum([child_criteria[criterion] for criterion in criteria])

                if cost + child_cost + self.heuristic(child) <= threshold:
                    frontier.put((cost + child_cost + self.heuristic(child), path + (child,), cost + child_cost))
                    reached.add(child)

            threshold_candidates = [(f + c) for f, _, c in frontier.queue]
            if threshold_candidates:
                threshold = min(threshold_candidates)
            else:
                threshold = float('inf')

        return list()
