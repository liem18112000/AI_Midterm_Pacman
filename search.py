"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
from time import sleep

from game import Directions

n = Directions.NORTH
s = Directions.SOUTH
e = Directions.EAST
w = Directions.WEST


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def depthFirstSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 05


def breadthFirstSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 06


def uniformCostSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 07


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# TODO 08 + 09
'''
students propose at least two heuristic functions for A*
'''

def foodHeuristic(state, problem, distanceFunction):
    """
    Heuristics = Distance of nearest food and pacman
    """
    pacman, foodGrid = state

    totalDis = 0
    pre_food = pacman
    for food in foodGrid.asList():
        cur_food = food
        totalDis += distanceFunction(pre_food, cur_food)
        pre_food = cur_food

    return totalDis

def sumHeuristic(state, problem, distanceFunction):
    pacman, foodGrid = state

    totalDis = 0
    for food in foodGrid.asList():
        totalDis += distanceFunction(pacman, food)

    return totalDis

def mixHeuristic(state, problem, distanceFunction):
    return (sumHeuristic(state, problem, distanceFunction) + foodHeuristic(state, problem, distanceFunction)) // 2

def aStarSearch(problem, heuristic=nullHeuristic, distanceFunction = None):
    '''
    return a path to the goal
    '''
    # TODO 10
    pqueue = util.PriorityQueue()

    visited = []

    pqueue.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem, distanceFunction) + 0)

    while not pqueue.isEmpty( ):
        curElement = pqueue.pop()
        vertex = curElement[0]
        resultPath = curElement[1]
        cost = curElement[2]

        # print(heuristic(vertex, problem, distanceFunction))

        # Check goal
        if problem.isGoalState(vertex):
            return resultPath

        # Adding new vertex to the visited
        visited.append(vertex)

        # Get all successors
        successors = problem.getSuccessors(vertex)

        # Go to next step
        for successor in successors:
            childVertex = successor[0]
            childPath = successor[1]
            childCost = successor[2]

            # Ignore visited nodes
            if childVertex not in visited:

                # Adding new vertex to the visited
                visited.append(childVertex)

                # Computing path of child vertex from start
                fullPath = resultPath + [childPath]

                # Computing culmulative backward cost of child vertex from start
                totalCost = cost + childCost

                # Pushing (Node, [Path], Culmulative backward cost) to the pqueue.
                pqueue.push((childVertex, fullPath, totalCost), totalCost + heuristic(childVertex, problem, distanceFunction))
            
            else:

                # Update (Node, [Path], Culmulative backward cost) in the pqueue.
                pqueue.update((childVertex, fullPath, totalCost), totalCost + heuristic(childVertex, problem, distanceFunction))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch