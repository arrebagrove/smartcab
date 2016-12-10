import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, take_random_action=False, alpha=0.1, gamma=0.5, eps=0.1):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # this variable indicates if we are going to take random actions instead of Q-learning
        self.take_random_action = take_random_action
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.num_success = 0
        self.tot_reward = 0

        # If we are going to take random actions, then we just ignore the alpha, gamma, epsilon params.
        if take_random_action:
            self.alpha = None
            self.gamma = None
            self.epsilon = None
        else:
            self.alpha = alpha
            self.gamma = gamma
            self.epsilon = eps
        self.Q = {}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.prev_state = None
        # TODO: Prepare for a new trip; reset any variables here, if required

    def get_random_action(self):
        valid_actions = [None, 'forward', 'left', 'right']
        r = random.randint(0,3)
        return valid_actions[r]

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # if we are taking only random actions, then no need to do Q-learning
        if self.take_random_action:
            action = self.get_random_action()
        else:
            ### Q-learning ####

            ## Define state ##
            # the current state is a tuple containing inputs, next way point and the deadline

            # A note regarding inputs : the other agents in the right do not have any impact on the action that our agent takes
            # for example, we can take a free right, dependin only upon the traffic_light and the oncoming and left coming agents
            # so we do not include the next way point of the right agents as part of our current state

            # we also consider the deadline as a binary variable, i.e whether more than 10 time steps away or less
            # based on whether there are more than 10 time steps away or less than that, the agent may choose different actions that are optimal.
            more_time_steps = deadline > 10
            self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint, more_time_steps)

            ## Select action according to your policy ##
            # Before selecting action, we first update the q-table for the prev state and prev action
            # After that we will figure out the optimal action from the current state
            #   Note tat the optimal action from the current state is nothing but the action which has max Q-value
            #   from the current state.

            # 1. update the Q-table
            # update the Q table backwards. i.e in the current state,
            # we will update the Q table for the prev state and prev action
            # we convert the
            if self.prev_state is not None:
                #prev_state_str = str(self.prev_state)
                if self.prev_state not in self.Q:
                    # since our Q table does not yet contain the state, we create the state
                    # and fill up with random values in [1,3] for each action from that state.
                    # ultimately after seeing many (s,a,r) tuples, these random values will
                    # converge to the true values
                    # note that we set initial weights uniformly in the range (1,3)
                    # so that we are above the reward that the act() fuction returns in some cases
                    # (note that that act() function can return rewards of -0.5, 0, 1, 2 until the destination is reached)
                    self.Q[self.prev_state] = {}
                    for action in Environment.valid_actions:
                        self.Q[self.prev_state][action] = random.uniform(1,3)
                prev_val = self.Q[self.prev_state][self.prev_action]

                # because the agent took the previous action from the prev state, it is now in the current state.
                # so to update the q-table for the prev (state, action) pair,
                # we use the following formula.
                if self.state not in self.Q:
                    self.Q[self.state] = {}
                    for action in Environment.valid_actions:
                        self.Q[self.state][action] = random.uniform(1,3)
                max_q_curr_state = max(self.Q[self.state].values())
                self.Q[self.prev_state][self.prev_action] = ((1-self.alpha) * prev_val) + \
                                                           (self.alpha * (self.prev_reward + self.gamma * max_q_curr_state))

            # 2. Select action
            # now that we have done one update to the Q table, we are back to business
            # let's find the action that has the max Q value for the current state and proceed with that action
            # with probability epsilon we choose a random action
            # and with probability 1-epsilon we choose the optimal action from the Q-table.
            if self.prev_state is None or random.random() < self.epsilon:
                action = None
                # we want an action that actually moves and explores (so we ignore None and retry)
                while action is None:
                    action = self.get_random_action()
            else:
                curr_state_dict = self.Q[self.state]
                action = max(curr_state_dict, key=curr_state_dict.get)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # and now, the curr state becomes the previous state
        #if self.prev_state is None:
        self.prev_state = self.state
        self.prev_action = action
        self.prev_reward = reward

        self.tot_reward += reward
        if(self.env.done):
            self.num_success += 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)

    kwargs = {'take_random_action': False, 'alpha': 0.3, 'gamma': 0.3, 'eps': 0.01}
    a = e.create_agent(LearningAgent, **kwargs)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    trials = 100
    sim.run(n_trials=trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print 'random_action={}, alpha={}, gamma={}, eps={}, success_rate={}, avg_reward={}'.format(
        a.take_random_action,
        a.alpha,
        a.gamma,
        a.epsilon,
        a.num_success * 1.0 / trials,
        a.tot_reward / trials)

def run1():
    for alpha in np.linspace(0.1, 0.9, num=5):
        for gamma in np.linspace(0.1, 0.9, num=5):
            for eps in np.linspace(0.01, 0.2, num=5):
                """Run the agent for a finite number of trials."""

                # Set up environment and agent
                e = Environment()  # create environment (also adds some dummy traffic)

                kwargs = {'take_random_action': False, 'alpha': alpha, 'gamma': gamma, 'eps': eps}
                a = e.create_agent(LearningAgent, **kwargs)  # create agent
                e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                # Now simulate it
                sim = Simulator(e, update_delay=0.0, display=True)  # create simulator (uses pygame when display=True, if available)
                # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                trials = 100
                sim.run(n_trials=trials)  # run for a specified number of trials
                # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

                print 'random_action={}, alpha={}, gamma={}, eps={}, success_rate={}, avg_reward={}'.format(
                    a.take_random_action,
                    a.alpha,
                    a.gamma,
                    a.epsilon,
                    a.num_success * 1.0 / trials,
                    a.tot_reward / trials)


if __name__ == '__main__':
    run()
