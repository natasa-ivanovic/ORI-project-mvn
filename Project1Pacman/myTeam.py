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
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    start = time.time()
    self.maxDepth = 4
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    successors = [(action, self.getSuccessor(gameState, action)) for action in actions]
    values = [(action, self.min_function(suc, 0, -99999, 99999)) for action, suc in successors]

    # max_val = values[0][1]
    # max_index = 0
    # for val in values:
    #   if val[1] > max_val:
    #     max_index = values.index(val)
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



  def min_function(self, gameState, depth, alpha, beta):
    if gameState.isOver():
      return self.getScore(gameState) - self.distances(gameState, self.index)
    else:
      opponenets = self.getOpponents(gameState)
      allGameStates = []
      self.foo(opponenets, 0, gameState, allGameStates)
      # print("Min agent 0 position:" + str(gameState.getAgentPosition(0)))
      # print("Min agent 1 position:" + str(gameState.getAgentPosition(1)))
      # print("Min agent 2 position:" + str(gameState.getAgentPosition(2)))
      # print("Min agent 3 position:" + str(gameState.getAgentPosition(3)))
      # print()
      if self.maxDepth // 2 > depth:
        allGameStates.sort(key=lambda state: self.distances(state, self.index))
      minEval = 99999
      for state in allGameStates:
        eval = self.max_function(state, depth, alpha, beta)
        minEval = min(minEval, eval)
        beta = min(beta, eval)
        if beta <= alpha:
          break

      return minEval

  def max_function(self, gameState, depth, alpha, beta):
    depth += 1
    if gameState.isOver() or self.maxDepth == depth:
      ret = self.getScore(gameState) - self.distances(gameState, self.index)
      return ret
    else:
      actions = gameState.getLegalActions(self.index)
      # print("Max agent 0 position:" + str(gameState.getAgentPosition(0)))
      # print("Max agent 1 position:" + str(gameState.getAgentPosition(1)))
      # print("Max agent 2 position:" + str(gameState.getAgentPosition(2)))
      # print("Max agent 3 position:" + str(gameState.getAgentPosition(3)))
      # print()
      successors = [self.getSuccessor(gameState, action) for action in actions]
      if self.maxDepth // 2 > depth:
        successors.sort(key=lambda state: - self.distances(state, self.index))
      maxEval = -99999
      for state in successors:
        eval = self.min_function(state, depth, alpha, beta)
        maxEval = max(maxEval, eval)
        alpha = max(alpha, eval)
        if beta <= alpha:
          break
      return maxEval


  def foo(self, opponents, idx, gameState, allGameStates):
    allActions = gameState.getLegalActions(opponents[idx])
    successors = [getFullPowerSuccessor(opponents[idx],gameState, action) for action in allActions]
    for suc in successors:
      if idx == len(opponents) - 1:
        allGameStates.append(suc)
      else:
        self.foo(opponents, idx+1, suc, allGameStates)

  def distances(self, gameState, idx):
    # myPos = gameState.getAgentState(self.index).getPosition()
    foodList = []
    if idx % 2 == 0:
      foodList = gameState.getBlueFood().asList()
    else:
      foodList = gameState.getRedFood().asList()
    ret = sum([self.getMazeDistance(gameState.getAgentPosition(idx), food) for food in foodList])
    return ret

def getFullPowerSuccessor(index, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(index, action)
    pos = successor.getAgentState(index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(index, action)
    else:
      return successor