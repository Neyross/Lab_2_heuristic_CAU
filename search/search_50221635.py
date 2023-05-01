from problem import RandomGeometricGraphProblem
from pathlib import Path
from typing import List, Tuple
from queue import PriorityQueue
from dataclasses import dataclass
from sys import maxsize


STUDENT_ID = Path(__file__).stem.split('_')[1]

@dataclass(order=True)
class State:
    cost: float  # cost of getting there without depth
    path: tuple

class Assignment:
    def __init__(self, problem: RandomGeometricGraphProblem):
        self.problem = problem

    def search(self, criteria: tuple, time_limit) -> List[str]:
        """
        Just a PriorityQueue that goes with the lowestcost for first criteria
        """
        frontier = PriorityQueue()
        frontier.put(State(0.0, (self.problem.initial_state,)))
        reached = [self.problem.initial_state]
        #stores the current lowest cost for a goal
        #goal = [State(float(maxsize), self.problem.initial_state)]
        while not frontier.empty():
            state = frontier.get()
            if self.problem.is_solution_path(state.path):
                return state.path
            childs = self.problem.expand(state.path[-1])
            for child, _ in childs:
                if child in reached:
                    continue
                current_cost = state.cost + self.cost_of(state.path[-1], child, criteria)
                frontier.put(State(current_cost, state.path + (child,)))
                reached.append(child)
        return []

    #get the cost of the action
    def cost_of(self, current, child, criteria):
        cost = 0
        if criteria[0] == "fee":
            if len(criteria) > 1:
                if criteria[1] == "road":
                    cost += self.problem.get_cost_of(current, child, criteria[1]) * 100
                else:
                    cost += self.problem.get_cost_of(current, child, criteria[1]) * 10
            else:
                cost += self.problem.get_cost_of(current, child, "road")
                cost += self.problem.get_cost_of(current, child, "time")
        cost += self.problem.get_cost_of(current, child, criteria[0])
        return cost
