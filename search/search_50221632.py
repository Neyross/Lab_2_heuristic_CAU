from problem import RandomGeometricGraphProblem
from pathlib import Path
from typing import List, Tuple
from queue import Queue


STUDENT_ID = Path(__file__).stem.split('_')[1]

class Assignment:
    def __init__(self, problem: RandomGeometricGraphProblem):
        self.problem = problem

    def search(self, criteria: tuple) -> List[str]:
        
        return list()
