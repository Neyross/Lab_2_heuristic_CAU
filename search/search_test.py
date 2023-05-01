from problem import RandomGeometricGraphProblem
from pathlib import Path
from typing import List, Tuple
from queue import PriorityQueue
from dataclasses import dataclass
from sys import maxsize


STUDENT_ID = Path(__file__).stem.split('_')[1]

@dataclass(order=True)
class State:
    score: float
    cost: float  # cost of getting there without heuristic
    path: tuple

class Assignment:
    def __init__(self, problem: RandomGeometricGraphProblem):
        self.problem = problem

    def heuristic(self, state: str) -> float:

        goal = self.problem.places_to_visit[-1]
        s1, s2 = self.problem.get_position_of(state)
        g1, g2 = self.problem.get_position_of(goal)

        h = abs(s1 - g1) + abs(s2 - g2)
        return h

    def search(self, criteria: tuple, time_limit) -> List[str]:
        """
        Just a PriorityQueue that goes with the lowestcost for first criteria
        """
        frontier = PriorityQueue()
        frontier.put(State(0.0, 0.0, (self.problem.initial_state,)))
        reached = [self.problem.initial_state]
        while not frontier.empty():
            state = frontier.get()
            childs = self.problem.expand(state.path[-1])
            reached.append(state.path[-1])
            for child, _ in childs:
                if self.problem.is_solution_path(state.path + (child,)):
                    return state.path + (child,)
                if child in reached:
                    continue
                current_cost = state.cost + self.cost_of(state.path[-1], child, criteria)
                frontier.put(State(current_cost + self.heuristic(child), current_cost , state.path + (child,)))
        return []

    #get the cost of the action
    def cost_of(self, current, child, criteria):
        cost = 0
        if criteria[0] == "fee":
            if len(criteria) > 1:
                if criteria[1] == "road":
                    cost += self.problem.get_cost_of(current, child, criteria[1]) * 10
                else:
                    cost += self.problem.get_cost_of(current, child, criteria[1]) * 1
            else:
                cost += self.problem.get_cost_of(current, child, "road") * 10
                cost += self.problem.get_cost_of(current, child, "time") * 1
            cost += self.problem.get_cost_of(current, child, criteria[0]) * 0.1
        elif criteria[0] == "road":
            cost += self.problem.get_cost_of(current, child, criteria[0]) * 10
        else:
            cost += self.problem.get_cost_of(current, child, criteria[0]) * 10
        return cost

    def child_in_queue(self, child, frontier: PriorityQueue, cost, path) -> bool:
        for i in frontier.queue:
            if i.path[-1] == child:
                if i.cost > cost:
                    del i
                    frontier.put(State(cost + self.heuristic(child), cost, path))
                    return True

        return False