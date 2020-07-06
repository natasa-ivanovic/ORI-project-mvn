# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import time

#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first = 'MiniMaxAgent', second = 'MiniMaxAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class MiniMaxAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''

    self.maxDepth = 2
    self.hasFoodWeight = gameState.data.layout.height + gameState.data.layout.width
    self.returnedFoodWeight = self.hasFoodWeight + 1
    self.opponentLimit = 2
    aggressive = self.index % 4 == 0
    if aggressive:
      self.mustGetOpponent = 100
      self.bestCarry = 1
    else:
      self.mustGetOpponent = 2
      self.bestCarry = 4
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    start = time.time()

    enemyDistance = [self.getMazeDistance(gameState.getAgentPosition(enemy), gameState.getAgentPosition(self.index)) for enemy in self.getOpponents(gameState)]

    if gameState.getAgentState(self.index).numCarrying < self.bestCarry:
      self.goPelletWeight = 1
      self.returnWeight = 0
    else:
      self.returnWeight = self.hasFoodWeight + 1
      self.goPelletWeight = 0

    if min(enemyDistance) < 2 and gameState.getAgentState(self.index).numCarrying != 0:
      self.returnWeight = self.hasFoodWeight + 1
      self.goPelletWeight = 0
    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      self.returnWeight = self.hasFoodWeight + 1
      self.goPelletWeight = 0
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)


    '''
    You should change this in your own agent.
    '''
    successors = [(action, self.getSuccessor(gameState, action)) for action in actions]
    values = []
    for action in actions :
      suc = self.getSuccessor(gameState, action)
      values.append((action, self.min_function(suc, 0, -99999, 99999, 0)))
    # print(self.index, "-", gameState.getAgentPosition(self.index),values)
    # print("______________________________________________________________________________________________________________")
    #
    # print("agent 0 position:" + str(gameState.getAgentPosition(0)))
    # print("agent 1 position:" + str(gameState.getAgentPosition(1)))
    # print("agent 2 position:" + str(gameState.getAgentPosition(2)))
    # print("agent 3 position:" + str(gameState.getAgentPosition(3)))


    print(time.time() - start)
    return max(values, key = lambda value: value[1])[0]

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor



  def min_function(self, gameState, depth, alpha, beta, agentIndex):
    if gameState.isOver():
      return self.getWeightedEstimates(gameState)
    opponenets = self.getOpponents(gameState)
    actions = gameState.getLegalActions(opponenets[agentIndex])
    minEval = 99999
    for action in actions:
      eval = 0
      succesor = gameState.generateSuccessor(opponenets[agentIndex], action)
      if len(opponenets) - 1 == agentIndex:
        eval = self.max_function(succesor, depth, alpha, beta)
      else:
        eval = self.min_function(succesor, depth, alpha, beta, agentIndex+1)
      minEval = min(minEval, eval)
      beta = min(beta, eval)
      if beta <= alpha:
        break

    return minEval

  def max_function(self, gameState, depth, alpha, beta):

    depth += 1
    if gameState.isOver() or self.maxDepth == depth:
      return self.getWeightedEstimates(gameState)

    actions = gameState.getLegalActions(self.index)

    # if self.maxDepth - 1 != depth:
    #    successors.sort(key=lambda state: - self.distances(state, self.index))
    maxEval = -99999
    for action in actions:

      state = self.getSuccessor(gameState, action)
      eval = self.min_function(state, depth, alpha, beta, 0)
      maxEval = max(maxEval, eval)
      alpha = max(alpha, eval)
      if beta <= alpha:
        break
    return maxEval

  def distances(self, gameState, idx):
    myPos = gameState.getAgentState(self.index).getPosition()
    foodList = self.getFood(gameState).asList()
    if(len(foodList) != 0)
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      return minDistance
    else:
      return 0

  def getWeightedEstimates(self, gameState):
    val1 = -self.goPelletWeight * self.distances(gameState, self.index)
    val2=      -self.returnWeight * self.getMazeDistance(gameState.getAgentPosition(self.index), (1,gameState.getAgentPosition(self.index)[1]))
    val3=      +self.pelletScore(gameState)
    val4=      -self.enemyAteMe(gameState)
    val5=      -self.eatEnemy(gameState)
    # print(val1, val2, val3, val4, val5)
    # print(val1 + val2 + val3 + val4 + val5)
    return val1 + val2 + val3 + val4 + val5

  def pelletScore(self, gameState):
    return self.getScore(gameState) * self.returnedFoodWeight \
           +gameState.getAgentState(self.index).numCarrying * self.hasFoodWeight

  def eatEnemy(self, gameState):
    sum = 0
    opponents = gameState.getBlueTeamIndices()
    me = gameState.getAgentPosition(self.index)
    for opponent in opponents:
      them = gameState.getAgentPosition(opponent)
      sum += gameState.getAgentState(opponent).numCarrying * self.mustGetOpponent * self.getMazeDistance(me, them)
    return sum

  def enemyAteMe(self, gameState):
    if self.start == gameState.getAgentPosition(self.index):
      return 1000
    return 0