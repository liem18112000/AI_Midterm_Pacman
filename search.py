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

def foodHeuristic(state, problem):
    """
    Heuristics = Distance of nearest food and pacman
    """
    pacman, foodGrid = state

    totalDis = 0
    pre_food = pacman
    for food in foodGrid.asList():
        cur_food = food
        totalDis += util.manhattanDistance(pre_food, cur_food)
        pre_food = cur_food

    return totalDis

def sumHeuristic(state, problem):
    pacman, foodGrid = state

    totalDis = 0
    for food in foodGrid.asList():
        totalDis += util.manhattanDistance(pacman, food)

    return totalDis

def ghostHeuristic(state, problem):
    pacman = state[0]
    AllGhostPos = problem.ghostPositions
    cost = 0
    for ghostPos in AllGhostPos:
        if abs(pacman[0] - ghostPos[0]) + abs(pacman[1] - ghostPos[1]) <= 2:
            cost = 99999
    return cost

def aStarSearch(problem, heuristic=nullHeuristic, maxThreshold = 999999):
    '''
    return a path to the goal
    '''
    # TODO 10
    pqueue = util.PriorityQueue()

    visited = []

    pqueue.push((problem.getStartState(), [], 0), heuristic(problem.getStartState(), problem) + 0)

    while not pqueue.isEmpty( ):
        curElement = pqueue.pop()
        vertex = curElement[0]
        resultPath = curElement[1]
        cost = curElement[2]

        # Check goal
        if problem.isGoalState(vertex):
            return resultPath
        elif cost + heuristic(vertex, problem) > maxThreshold:
            return cost + heuristic(vertex, problem)

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
                pqueue.push((childVertex, fullPath, totalCost), totalCost + heuristic(childVertex, problem))
            
            else:

                # Update (Node, [Path], Culmulative backward cost) in the pqueue.
                pqueue.update((childVertex, fullPath, totalCost), totalCost + heuristic(childVertex, problem))


def IDA(problem, heuristic=nullHeuristic):
    threshold = heuristic(problem.getStartState(), problem)

    while True:
        result = aStarSearch(problem, heuristic, threshold)
        if isinstance(result,int):
            threshold = result
        else:
            return result

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch