# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

__author__ = "Sike Ogieva '25 and Lindsay Ward '25"

import util
import math


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


class Node():
    """
        a Node object has: a state pointer, total path cost,
        last action taken and parent Node.
        It stores a more comprehensive idea of what leads to
        a state
    """

    def __init__(self, state, pathCost, lastAction, parentNode):
        self.state = state  # tuple
        self.pathCost = pathCost  # int
        self.lastAction = lastAction  # string
        self.parentNode = parentNode  # Node

    def getState(self):
        return self.state

    def getPathCost(self):
        return self.pathCost

    def getLastAction(self):
        return self.lastAction

    def getParentNode(self):
        return self.parentNode


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm
    (i.e. maintain an explored set)

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
        # Start: (5, 5)
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        # Is the start a goal? False
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
        # Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]

    """

    # inits
    frontier = util.Stack()  # LIFO
    # new node = state, cost, action, parent
    frontier.push(Node(problem.getStartState(), 0, None, None))
    explored = set()

    # until we find the goal state or run out of unexplored nodes, do:
    while True:
        # failure, we ran out of unexplored nodes without finding a solution
        if frontier.isEmpty():
            return []

        # update current node and its state
        presentNode = frontier.pop()  # chooses the deepest node
        presentState = presentNode.getState()

        # did we find the solution? return its path
        if problem.isGoalState(presentState):
            return getPath(presentNode, problem.getStartState())

        if presentState not in explored:
            for succession in problem.getSuccessors(presentState):
                # new node = state, cost, action, parent
                childNode = Node(succession[0], succession[2] + presentNode.getPathCost(), succession[1], presentNode)

                ''' why do we check the explored set, but not the frontier for visited states? 
                    the frontier - a stack - is non-iterable. any better reason?
                '''
                if succession[0] not in explored:
                    frontier.push(childNode)

        ''' Notice we add states, not nodes to the explored set? 
        The pseudocode was misleading on this. Repeated states are very common,
        and are the time-chomping problem, not repeated nodes.
        Moreover, repeated nodes are included in repeated states.'''
        explored.add(presentState)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first: level-order traversal
        See dfs for detailed comments
    """
    frontier = util.Queue()  # FIFO
    frontier.push(Node(problem.getStartState(), 0, None, None))
    explored = set()

    while True:
        if frontier.isEmpty():
            return []

        presentNode = frontier.pop()  # chooses the shallowest node
        presentState = presentNode.getState()

        if problem.isGoalState(presentState):
            return getPath(presentNode, problem.getStartState())

        if presentState not in explored:
            for succession in problem.getSuccessors(presentState):
                childNode = Node(succession[0], succession[2] + presentNode.getPathCost(), succession[1], presentNode)
                if succession[0] not in explored:
                    frontier.push(childNode)

        explored.add(presentState)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    frontier = util.PriorityQueue()  # lowest cost -> highest priority
    frontier.push(Node(problem.getStartState(), 0, None, None), 0)
    explored = set()

    while True:
        if frontier.isEmpty():
            return []

        presentNode = frontier.pop()  # chooses the cheapest node
        presentState = presentNode.getState()

        if problem.isGoalState(presentState):
            return getPath(presentNode, problem.getStartState())

        if presentState not in explored:
            for succession in problem.getSuccessors(presentState):
                childNode = Node(succession[0], succession[2] + presentNode.getPathCost(), succession[1], presentNode)
                if succession[0] not in explored:
                    frontier.push(childNode, childNode.getPathCost())

        explored.add(presentState)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    frontier = util.PriorityQueue()  # lowest (cost+heuristic) -> highest priority

    # priority = 0 (cost) + heuristic(problem.getStartState(), problem)
    frontier.push(Node(problem.getStartState(), 0, None, None), heuristic(problem.getStartState(), problem))
    explored = set()

    while True:
        if frontier.isEmpty():
            return []

        presentNode = frontier.pop()  # chooses the cheapest node
        presentState = presentNode.getState()

        if problem.isGoalState(presentState):
            return getPath(presentNode, problem.getStartState())

        if presentState not in explored:
            for succession in problem.getSuccessors(presentState):
                childNode = Node(succession[0], succession[2] + presentNode.getPathCost(), succession[1], presentNode)
                if succession[0] not in explored:
                    frontier.push(childNode, childNode.getPathCost() + heuristic(succession[0], problem))

        explored.add(presentState)


def getPath(node, state):
    """
    Given a Node, this function recreates the path of Actions that led to it from the start Node.
    """
    path = []
    while node.getState() != state:
        path.append(node.getLastAction())
        node = node.getParentNode()
    return path[::-1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
