import gym
from gym import spaces
import random

# Custom Gym-like environment for the probabilistic planning domain
class ProbabilisticPlanningDomain(gym.Env):
    def __init__(self):
        super().__init__()
        # Define the states and actions
        self.states = ['d1', 'd2', 'd3', 'd4', 'd5']
        self.actions = ['m12', 'm14', 'm21', 'm23', 'm32', 'm34', 'm41', 'm43', 'm45', 'm52', 'm54']

        # Start and goal states
        self.start_state = 'd1'
        self.goal_state = 'd4'.strip()
        self.state = self.start_state

        # Action space and observation space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(len(self.states))

        self._init_safe_policy = {'d1':'m14','d2':'m23','d3':'m34','d4':None,'d5':'m54'}

        self.transitions = {
            ('d1', 'm12'): [('d2', 1.0)],
            ('d1', 'm14'): [('d4', 0.5), ('d1', 0.5)],
            ('d2', 'm21'): [('d1', 1.0)],
            ('d2', 'm23'): [('d3', 0.8), ('d5', 0.2)],
            ('d3', 'm32'): [('d2', 1.0)],
            ('d3', 'm34'): [('d4', 1.0)],
            ('d4', 'm41'): [('d1', 1.0)],
            ('d4', 'm43'): [('d3', 1.0)],
            ('d4', 'm45'): [('d5', 1.0)],
            ('d5', 'm54'): [('d4', 1.0)],
            ('d5', 'm52'): [('d2', 1.0)],
        }

        self.Pr = {key + (target,): prob 
                   for key, transitions in self.transitions.items() 
                   for target, prob in transitions}

        self.applicable_actions = {
            'd1': ['m12', 'm14'],
            'd2': ['m23', 'm21'],
            'd3': ['m34', 'm32'],
            'd4': ['m45', 'm41', 'm43'],
            'd5': ['m54', 'm52']
        }

        self._gamma = {
            ('d1', 'm12'): ['d2'],
            ('d1', 'm14'): ['d4', 'd1'],
            ('d2', 'm21'): ['d1'],
            ('d2', 'm23'): ['d3', 'd5'],
            ('d3', 'm32'): ['d2'],
            ('d3', 'm34'): ['d4'],
            ('d4', 'm41'): ['d1'],
            ('d4', 'm43'): ['d3'],
            ('d4', 'm45'): ['d5'],
            ('d5', 'm54'): ['d4'],
            ('d5', 'm52'): ['d2'],
        }

        self.costs = {
            ('d1', 'm12', 'd2'): -100,
            ('d1', 'm14', 'd4'): -80,
            ('d1', 'm14', 'd1'): -80,
            ('d2', 'm21', 'd1'): -100,
            ('d2', 'm23', 'd3'): -1,
            ('d2', 'm23', 'd5'): -1,
            ('d3', 'm32', 'd2'): -1,
            ('d3', 'm34', 'd4'): -1,
            ('d4', 'm43', 'd3'): -1,
            ('d4', 'm45', 'd5'): -1,
            ('d4', 'm41', 'd1'): -80,
            ('d5', 'm54', 'd4'): -1,
            ('d5', 'm52', 'd2'): -1
        }

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        if isinstance(action, str):
            action_name = action
        else:
            action_name = self.actions[action]
        possible_transitions = self.transitions.get((self.state, action_name), [])

        if not possible_transitions:
            # Invalid action, return current state and large cost
            return self.state, 10, False, {}

        # Sample the next state based on probabilities
        next_states, probs = zip(*possible_transitions)
        next_state = random.choices(next_states, weights=probs)[0]

        # Check if goal state is reached
        next_state = next_state.strip()
        cost = self.get_cost(self.state, action_name, next_state)
        done = next_state == self.S_g()
        self.state = next_state  
        return self.state, cost, done, {}

    def render(self, mode='human'):
        print(f"Current state: {self.state}") 
    def S_g(self):
        return self.goal_state
    def Applicable(self, s):
        return self.applicable_actions[s]
    def get_cost(self, s, a, s_prime):
        return self.costs[(s, a, s_prime)]
    def gamma(self, state, action):
        if action is None:
            return []
        return self._gamma[(state, action)]
    def init_safe_policy(self):
        return self._init_safe_policy
