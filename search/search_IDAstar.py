from problem import RandomGeometricGraphProblem
from pathlib import Path
from typing import List
from queue import PriorityQueue


STUDENT_ID = Path(__file__).stem.split('_')[1]


class Assignment:
    def __init__(self, problem: RandomGeometricGraphProblem):
        self.problem = problem
        self.heuristic_cache = {}

    def heuristic(self, state: str) -> float:
        if state in self.heuristic_cache:
            return self.heuristic_cache[state]

        goal = self.problem.places_to_visit[-1]
        s1, s2 = self.problem.get_position_of(state)
        g1, g2 = self.problem.get_position_of(goal)

        h = abs(s1 - g1) + abs(s2 - g2)
        self.heuristic_cache[state] = h
        return h

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
        frontier.put((self.heuristic(self.problem.initial_state),
                     (self.problem.initial_state,), 0))

        reached = set({(self.problem.initial_state,)})

        while not frontier.empty():
            # get the path with the lowest cost
            _, path, cost = frontier.get()
            # if the path is a solution, return it
            if self.problem.is_solution_path(path):
                return path

            # for each child of the last node in the path
            nodes = self.problem.expand(path[-1])
            for child, _ in nodes:
                # if the child has already been visited, continue to the next child
                if child in reached:
                    continue

                # calculate the cost of the child
                c_heuristic = self.heuristic(child)
                child_cost = sum(self.problem.get_cost_of(
                    path[-1], child, criter) for criter in criteria)

                # if the cost of the child is less than the threshold
                if cost + child_cost + c_heuristic <= threshold:
                    frontier.put((cost + child_cost + c_heuristic,
                                  path + (child,),
                                  cost + child_cost))
                    reached.add(child)

            # get the threshold candidates from the frontier
            threshold_candidates = [(f + c) for f, _, c in frontier.queue]
            # if there are threshold candidates, set the threshold to the minimum of them
            threshold = min(threshold_candidates) if threshold_candidates else float('inf')


        # if the frontier is empty, return an empty list
        return list()
