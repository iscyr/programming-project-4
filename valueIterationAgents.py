# valueIterationAgents.py
# -----------------------
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








# valueIterationAgents.py
# -----------------------
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








import mdp, util




from learningAgents import ValueEstimationAgent
import collections
import random




class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*




        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.




          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()




    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            #duplicate the previous iterations values
            newVals = self.values.copy()


            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newVals[state] = 0
                    continue
                actions = self.mdp.getPossibleActions(state)
                if not actions:
                    continue


                qVals = [self.computeQValueFromValues(state, action) for action in actions]
                newVals[state] = max(qVals)
           
            self.values = newVals










    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]








    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #set variables
        qval = 0.0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)




        for next_state, probability in transitions:
            reward = self.mdp.getReward(state, action, next_state)
            futureVal = self.values[next_state]
            qval += probability * (reward + self.discount * futureVal)




        return qval












        #util.raiseNotDefined()




    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.




          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"




        moves = self.mdp.getPossibleActions(state)
        if not moves:
            return None
       
        qval = [(move, self.computeQValueFromValues(state, move)) for move in moves]
        maxVal = max(q for _, q in qval)




        bestMoves = [move for move, q in qval if q == maxVal]




        return random.choice(bestMoves)
















        util.raiseNotDefined()




    def getPolicy(self, state):
        return self.computeActionFromValues(state)




    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)




    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)




class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*




        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.




          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)




    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            # Determine the state to update in this iteration
            state = states[i % len(states)]
            
            # Skip terminal states
            if self.mdp.isTerminal(state):
                continue
            
            # Compute the maximum Q-value for the state
            possibleActions = self.mdp.getPossibleActions(state)
            maxQValue = float('-inf')
            for action in possibleActions:
                qValue = self.computeQValueFromValues(state, action)
                maxQValue = max(maxQValue, qValue)
            
            # Update the value of the state
            self.values[state] = maxQValue




class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*




        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)




    def runValueIteration(self):
        # Step 1: Compute predecessors of all states
        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
            predecessors[state] = set()
        for state in states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            predecessors[nextState].add(state)

        # Step 2: Initialize an empty priority queue
        priorityQueue = util.PriorityQueue()

        # Step 3: Compute initial priorities for all non-terminal states
        for state in states:
            if not self.mdp.isTerminal(state):
                # Compute the highest Q-value for the state
                possibleActions = self.mdp.getPossibleActions(state)
                maxQValue = max(self.computeQValueFromValues(state, action) for action in possibleActions)
                diff = abs(self.values[state] - maxQValue)
                # Push the state into the priority queue with priority -diff
                priorityQueue.push(state, -diff)

        # Step 4: Perform value iteration for the specified number of iterations
        for _ in range(self.iterations):
            if priorityQueue.isEmpty():
                break

            # Pop a state with the highest priority (lowest -diff)
            state = priorityQueue.pop()

            # Update the value of the state if it is not terminal
            if not self.mdp.isTerminal(state):
                possibleActions = self.mdp.getPossibleActions(state)
                maxQValue = max(self.computeQValueFromValues(state, action) for action in possibleActions)
                self.values[state] = maxQValue

            # Update the priorities of the predecessors of the state
            for predecessor in predecessors[state]:
                if not self.mdp.isTerminal(predecessor):
                    possibleActions = self.mdp.getPossibleActions(predecessor)
                    maxQValue = max(self.computeQValueFromValues(predecessor, action) for action in possibleActions)
                    diff = abs(self.values[predecessor] - maxQValue)
                    # Push the predecessor into the priority queue if diff > theta
                    if diff > self.theta:
                        priorityQueue.update(predecessor, -diff)



