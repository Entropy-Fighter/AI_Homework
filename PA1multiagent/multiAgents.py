# multiAgents.py
# --------------
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
#import numpy as np

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print(newGhostStates)
        "*** YOUR CODE HERE ***"
       
        for ghost_state in newGhostStates:
            if newPos == ghost_state.getPosition() and min(newScaredTimes) == 0:
                return -1
            
        oldFood = currentGameState.getFood()
        for food in oldFood.asList():
            if newPos == food:
                return 1
        
        closest_dis_f = 99999
        for food in newFood.asList():
            if util.manhattanDistance(food, newPos) < closest_dis_f:
                closest_dis_f = util.manhattanDistance(food, newPos)
                
        closest_dis_g = 99999
        for ghost in newGhostStates:
            if util.manhattanDistance(ghost.getPosition(), newPos) < closest_dis_g:
                closest_dis_g = util.manhattanDistance(ghost.getPosition(), newPos)
        
        # print
        return (-closest_dis_f + closest_dis_g) / max(closest_dis_g, closest_dis_f)
        # return childGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # print(gameState.getLegalActions(0))
        # util.raiseNotDefined()
        def value(state, tree_depth):
            if tree_depth == self.depth * state.getNumAgents() or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if tree_depth % state.getNumAgents() == 0:
                return max_value(state, tree_depth)
            else:
                return min_value(state, tree_depth)
                    
        def max_value(state, tree_depth):
            v = -99999
            for action in state.getLegalActions(0):
                v = max(v, value(state.getNextState(0, action), tree_depth + 1))
            return v
        def min_value(state, tree_depth):
            v = 99999
            for action in state.getLegalActions(tree_depth % state.getNumAgents()):
                v = min(v, value(state.getNextState(tree_depth % state.getNumAgents(), action), tree_depth + 1))
            return v
           
        v_list = [] 
        for action in gameState.getLegalActions(0):
            v_list.append(value(gameState.getNextState(0, action), 1))
        return gameState.getLegalActions(0)[v_list.index(max(v_list))]
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def value(state, tree_depth, a, b):
            if tree_depth == self.depth * state.getNumAgents() or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if tree_depth % state.getNumAgents() == 0:
                return max_value(state, tree_depth, a, b)
            else:
                return min_value(state, tree_depth, a, b)
                    
        def max_value(state, tree_depth, a, b):
            v = -99999
            for action in state.getLegalActions(0):
                v = max(v, value(state.getNextState(0, action), tree_depth + 1, a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v
        def min_value(state, tree_depth, a, b):
            v = 99999
            for action in state.getLegalActions(tree_depth % state.getNumAgents()):
                v = min(v, value(state.getNextState(tree_depth % state.getNumAgents(), action), tree_depth + 1, a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v
           
        v = -99999 
        a = -999999
        b = 999999
        i = 0
        for index, action in enumerate(gameState.getLegalActions(0)):
            if v < value(gameState.getNextState(0, action), 1, a, b):
                v = value(gameState.getNextState(0, action), 1, a, b)
                i = index
            if v > b:
                i = index
                break
            a = max(v, a)
        return gameState.getLegalActions(0)[i]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(state, tree_depth):
            if tree_depth == self.depth * state.getNumAgents() or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            if tree_depth % state.getNumAgents() == 0:
                return max_value(state, tree_depth)
            else:
                return min_value(state, tree_depth)
                    
        def max_value(state, tree_depth):
            v = -99999
            for action in state.getLegalActions(0):
                v = max(v, value(state.getNextState(0, action), tree_depth + 1))
            return v
        def min_value(state, tree_depth): # use the mean value
            v = 0
            for action in state.getLegalActions(tree_depth % state.getNumAgents()):
                v = v + value(state.getNextState(tree_depth % state.getNumAgents(), action), tree_depth + 1)
            return v / len(state.getLegalActions(tree_depth % state.getNumAgents()))
           
        v_list = [] 
        for action in gameState.getLegalActions(0):
            v_list.append(value(gameState.getNextState(0, action), 1))
        return gameState.getLegalActions(0)[v_list.index(max(v_list))]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Firstly, we consider that the ghosts are scared, in this situation,
    our pacman should be brave enough to get close to them and eat them.
    Secondly, we let the pacman to find the closest capsule.
    Lastly, we apply the same strategy as problem 1(of course do some slight changes).
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()

                
    closest_dis_g = 99999
    for ghost in newGhostStates:
        if util.manhattanDistance(ghost.getPosition(), newPos) < closest_dis_g:
            closest_dis_g = util.manhattanDistance(ghost.getPosition(), newPos)
        
    if max(newScaredTimes) > 0:
        return scoreEvaluationFunction(currentGameState) + max(newScaredTimes) / (closest_dis_g + 0.1)
    
    closest_dis_c = 99999
    for c in newCapsules:
        if util.manhattanDistance(c, newPos) < closest_dis_c:
            closest_dis_c = util.manhattanDistance(c, newPos)
            
    if len(newCapsules) != 0:
        return scoreEvaluationFunction(currentGameState) - closest_dis_c
    
    if currentGameState.isLose():
        return -10
    for ghost_state in newGhostStates:
        if newPos == ghost_state.getPosition() and min(newScaredTimes) == 0:
            return -10
            
    oldFood = currentGameState.getFood()
    for food in oldFood.asList():
        if newPos == food:
            return scoreEvaluationFunction(currentGameState)
    
    
    closest_dis_f = 99999
    for food in newFood.asList():
        if util.manhattanDistance(food, newPos) < closest_dis_f:
            closest_dis_f = util.manhattanDistance(food, newPos)
    
    return scoreEvaluationFunction(currentGameState) + (-closest_dis_f + closest_dis_g) / max(closest_dis_g, closest_dis_f)
    # return childGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
#         def value(state, tree_depth):
#             if tree_depth == self.depth * state.getNumAgents() or state.isWin() or state.isLose():
#                 return efunc(state)
#             if tree_depth % state.getNumAgents() == 0:
#                 return max_value(state, tree_depth)
#             else:
#                 return min_value(state, tree_depth)
                    
#         def max_value(state, tree_depth):
#             v = -99999
#             for action in state.getLegalActions(0):
#                 v = max(v, value(state.getNextState(0, action), tree_depth + 1))
#             return v
#         def min_value(state, tree_depth):
#             v = 99999
#             for action in state.getLegalActions(tree_depth % state.getNumAgents()):
#                 v = min(v, value(state.getNextState(tree_depth % state.getNumAgents(), action), tree_depth + 1))
#             return v
           
#         v_list = [] 
#         for action in gameState.getLegalActions(0):
#             v_list.append(value(gameState.getNextState(0, action), 1))
#         return gameState.getLegalActions(0)[v_list.index(max(v_list))]
    
# def efunc(currentGameState):
#     """
#     Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
#     evaluation function (question 5).

#     DESCRIPTION: <write something here so we know what you did>
#     """
#     "*** YOUR CODE HERE ***"
#     #util.raiseNotDefined()

#     newPos = currentGameState.getPacmanPosition()
#     newFood = currentGameState.getFood()
#     newGhostStates = currentGameState.getGhostStates()
#     newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
#     newCapsules = currentGameState.getCapsules()

                
#     closest_dis_g = 99999
#     for ghost in newGhostStates:
#         if util.manhattanDistance(ghost.getPosition(), newPos) < closest_dis_g:
#             closest_dis_g = util.manhattanDistance(ghost.getPosition(), newPos)
        
#     if max(newScaredTimes) > 0:
#         return scoreEvaluationFunction(currentGameState) + max(newScaredTimes) / (closest_dis_g + 0.1)
    
#     closest_dis_c = 99999
#     for c in newCapsules:
#         if util.manhattanDistance(c, newPos) < closest_dis_c:
#             closest_dis_c = util.manhattanDistance(c, newPos)
            
#     if len(newCapsules) != 0:
#         return scoreEvaluationFunction(currentGameState) - closest_dis_c
    
#     if currentGameState.isLose():
#         return -10
#     for ghost_state in newGhostStates:
#         if newPos == ghost_state.getPosition() and min(newScaredTimes) == 0:
#             return -10
            
#     oldFood = currentGameState.getFood()
#     for food in oldFood.asList():
#         if newPos == food:
#             return scoreEvaluationFunction(currentGameState)
    
    
#     closest_dis_f = 99999
#     for food in newFood.asList():
#         if util.manhattanDistance(food, newPos) < closest_dis_f:
#             closest_dis_f = util.manhattanDistance(food, newPos)
    
#     return scoreEvaluationFunction(currentGameState) + (-closest_dis_f + closest_dis_g) / max(closest_dis_g, closest_dis_f)