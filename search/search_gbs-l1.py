from problem import RandomGeometricGraphProblem
from pathlib import Path
from typing import List, Tuple
from queue import Queue, PriorityQueue


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
        Your documentation should contain the following things:
        - Which algorithm that you designed?
        - Why did you select it?
        - What does each command do? (line-by-line documentation)
        """
        frontier = PriorityQueue()
        frontier.put((self.heuristic(self.problem.initial_state),
                      (self.problem.initial_state,)))
        reached = [self.problem.initial_state]

        while not frontier.empty():
            _, path = frontier.get()
            if self.problem.is_solution_path(path):
                return list(path)

            for child, _ in self.problem.expand(path[-1]):
                if child in reached:
                    continue
                frontier.put((self.heuristic(child),
                              path + (child,)))
                reached.append(child)

        return []
