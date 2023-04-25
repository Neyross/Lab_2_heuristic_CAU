from problem import RandomGeometricGraphProblem
from pathlib import Path
from typing import List, Tuple
from queue import PriorityQueue
from dataclasses import dataclass
from sys import maxsize


STUDENT_ID = Path(__file__).stem.split('_')[1]

@dataclass(order=True)
class State:
    cost: float
    path: tuple

class Assignment:
    def __init__(self, problem: RandomGeometricGraphProblem):
        self.problem = problem

    def search(self, criteria: tuple) -> List[str]:
        """
        Just a PriorityQueue that goes with the lowest added cost of criterias
        """
        frontier = PriorityQueue()
        frontier.put(State(0.0, (self.problem.initial_state,)))
        reached = [self.problem.initial_state]
        #stores the current lowest cost for a goal
        goal = [State(float(maxsize), self.problem.initial_state)]
        while not frontier.empty():
            state = frontier.get()
            if self.problem.is_solution_path(state.path):
                if state.cost < goal[0].cost:
                    goal[0] = state
            childs = self.problem.expand(state.path[-1])
            for child, _ in childs:
                if child in reached:
                    continue
                # get the added cost of the state ie if criteria road and fee get road + fee cost
                current_cost = state.cost + self.cost_of(state.path[-1], child, criteria)
                frontier.put(State(current_cost, state.path + (child,)))
                reached.append(child)
        return goal[0].path

    #get the cost of the action
    def cost_of(self, current, child, criteria):
        cost = 0
        for criter in criteria:
            cost += self.problem.get_cost_of(current, child, criter)
        return cost

