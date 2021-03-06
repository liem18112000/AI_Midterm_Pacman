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
    start = problem.getStartState()
    stack = util.Stack()
    visited = []
    print problem.getSuccessors(start)
    stack.push(start)

    # Create a back tracking table as a dictionary with Key = current Node , value = (father node, directon)
    backtrack = dict()
    # Init backtracking value for start node
    backtrack[str(start[0])] = (None, None)

    i = 0
    while stack:

        curNode = stack.pop()               # Taking out node for examine
        visited.append(curNode[0])          # Mark it as visited

        # print curNode
        #print curNode[0]
        # print stack.list
        #print visited
        if (problem.isGoalState(curNode)):                  # Goal condition
            #print "Found Goalllll"
            # DFS_backtracking take in goal and backtrack table and return array of paths
            path = DFS_backtracking(curNode[0], backtrack)
            return path

        successors = problem.getSuccessors(curNode)
        for childNode in successors:                        # Extract successors nodes
            if childNode[0][0] not in visited:
                # If it is not visted then put in the stack
                stack.push(childNode[0])
                # Add its origin information to the backtracking table
                backtrack[str(childNode[0][0])] = (curNode[0], childNode[1])
                #print childNode[0][0]
                #print backtrack

        # print '\n'

# This method return the path from start node to goal


def DFS_backtracking(goal, backtrack):
    path = []

    cur = goal
    from_node = backtrack[str(cur)][0]
    direction = backtrack[str(cur)][1]
    path.append(backtrack[str(cur)][1])
    while True:
        cur = from_node
        from_node = backtrack[str(cur)][0]
        if from_node == None:
            break

        direction = backtrack[str(cur)][1]
        path.insert(0, direction)
    return path


def depthFirstSearch_Multi(problem):
    '''
    return a path to all the goals
    '''
    # TODO 11
    start = problem.getStartState()

    stack = util.Stack()                    # Init stack
    visited = []                            # Init visited
    # Init backtrack table to store expansion information to a single food
    backtrack = dict()
    backtrack_multi = []                    # Init list to store a list of path

    stack.push(start)                       # Push a start node in the stack
    # Init backtracking value for start node
    backtrack[str(start[0])] = (None, None)
    food_remain = start[1].count()          # Mark it as visted
    while stack.list:

        curNode = stack.pop()               # Taking out node for examine
        visited.append(curNode[0])          # Mark it as visited

        # If a food is eaten , reset the DFS
        if (curNode[1].count() < food_remain):
            print "Found Food!!!!!!!!!"
            food_remain = curNode[1].count()
            stack.list = []                                      # reset stack
            visited = []                                         # reset visted
            # Store a path from 1 start to 1 goal
            backtrack_multi.append((backtrack, (curNode[0])))
            # from here ,perform a new DFS search with start node is the last eaten food node
            backtrack = {}
            # Set the current goal as start node for the next DFS search
            backtrack[str(curNode[0])] = (None, None)
            # mark the it as visited
            visited.append(curNode[0])
            if food_remain == 0:  # stop the search when all foods are eaten
                print "Eaten all foods!!!!!!!"
                break

        successors = problem.getSuccessors(curNode)
        for childNode in successors:                        # Extract successors nodes
            if childNode[0][0] not in visited:
                # If it is not visted then put in the stack
                stack.push(childNode[0])
                # Add its origin information to the backtracking table
                backtrack[str(childNode[0][0])] = (curNode[0], childNode[1])

    path = []                               # Init the full path
    for element in backtrack_multi:         # Attach the path to single food to the full path
        goal = element[1]                   # Take out goal coordinate
        bt_tbl = element[0]                 # Take out backtrack table
        # Return a path to a single food
        single_path = DFS_backtracking(goal, bt_tbl)
        # Add new path to the end of the full path
        path.extend(single_path)

    return path


def breadthFirstSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 06
    queue = util.Queue()
    visited = []
    queue.push((problem.getStartState(), []))
    while not queue.isEmpty():
        cur_node, cur_path = queue.pop()
        if problem.isGoalState():
            return cur_path
        else:
            if cur_node not in visited:
                visited.append(cur_node)
                for child in problem.getSuccessors(cur_node):
                    child_node, child_path = child
                    fullPath = cur_path + [child_path]
                    queue.push((child_node, fullPath))


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

    while not pqueue.isEmpty():
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
