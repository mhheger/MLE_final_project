import os
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .model import DQN
from .features import coin_feature, mortality, bomb_efficiency, calculate_directional_feature

from collections import namedtuple, deque

DeepQNet = namedtuple('DeepQNet', ('policy_net', 'target_net', 'epochs'))


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
SIZE = 9
INPUT_SIZE = 5*SIZE*SIZE+1+1+4
P_START = 0.9 
P_END = 0.1
RATE = 0.0001
RATE2 = 0
def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    global device
    if self.train:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
            )
    else: 
        device = "cpu"


    if not os.path.isfile("renate-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = DeepQNet(DQN(INPUT_SIZE, 6), DQN(INPUT_SIZE, 6), [0])
    else:
        self.logger.info("Loading model from saved state.")
        with open("renate-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.model.policy_net.to(device)
    self.model.target_net.to(device)
    self.stats = {
        "steps_survived": 0,
        "coins_collected": 0, 
        "crates_destroyed":0, 
        "opponents_killed": 0, 
        "invalid_actions": 0,
        "score": 1, 
        "total_reward": 0,
        "mean_loss": 0
    }
    self.loss = []
    self.pos_history = deque(maxlen = 20 )
    self.coordinate_history = deque([], 20)
    self.current_round = 0



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    state = torch.tensor(state_to_features(game_state), dtype= torch.float32, device = device)
    round = self.model.epochs[0]
    self.pos_history.append(game_state["self"][3])

    random_prob = P_END+ (P_START-P_END)*np.exp(-RATE*round*(np.exp(RATE2*game_state["round"])))
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        choice = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        #choice = acting_rules(game_state)
        return(choice)

    self.logger.debug("Querying model for action.")
    if self.pos_history.count(game_state["self"][3]) > 8: 
        valid_actions = get_valid_actions(self, game_state)
        if valid_actions:
            return(np.random.choice(valid_actions))
    #if not self.train and random.random() < P_END:
       # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    #print(self.model.policy_net.forward(state).detach().cpu().numpy())
    #index =np.argmax(self.model.policy_net.forward(state).detach().cpu().numpy())
    #print(ACTIONS[index])
    #return ACTIONS[index]

    p = F.softmax(self.model.policy_net.forward(state).detach()).cpu().numpy()
    return np.random.choice(ACTIONS, p=p)


def state_to_features(game_state: dict, rotate = 0) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    coin_card = np.zeros_like(game_state["field"])
    bomb_card = mortality(game_state["bombs"],game_state["field"])
    opponent_card = np.zeros_like(game_state["field"])
    for item in game_state["coins"]:
        coin_card[item[0], item[1]]=10
    for agent in game_state["others"]:
        _,_,_,(x,y) = agent
        opponent_card[x,y] = 10
    coin_field  = np.zeros((34,34))
    coin_field[8:25, 8:25] = coin_card
    opponent_field = np.zeros((34,34))
    opponent_field[8:25, 8:25] = opponent_card
    bomb_field  = np.zeros((34,34))
    bomb_field[8:25, 8:25] = bomb_card
    explosion_field  = np.zeros((34,34))
    explosion_field[8:25, 8:25] = game_state["explosion_map"]
    field = -1*np.ones((34,34))
    field[8:25, 8:25] = game_state["field"]
    posx = game_state["self"][3][0]
    posy = game_state["self"][3][1]

    bomb_available = game_state["self"][2]
    bomb_effect = bomb_efficiency(opponent_field[8:25, 8:25]+field[8:25, 8:25], posx, posy)


    low = int(np.floor(SIZE/2))
    x_low = posx+8-low
    x_up = posx+8+SIZE-low
    y_low = posy+8-low
    y_up = posy+8+SIZE-low

    coin_field_centered = coin_field[x_low:x_up, y_low:y_up]
    field_centered = field[x_low:x_up, y_low:y_up]
    bomb_field_centered = bomb_field[x_low:x_up, y_low:y_up]
    opponent_field_centered = opponent_field[x_low:x_up, y_low:y_up]
    explosion_field_centered = explosion_field[x_low:x_up, y_low:y_up]
    best_coin_direction= coin_feature(coin_card, game_state["field"], (posx,posy))
    directional_feature = calculate_directional_feature(coin_field[posx:posx+17,posy:posy+17],field[posx:posx+17,posy:posy+17])
                
    for i in np.arange(rotate):
        coin_field_centered = np.rot90(coin_field_centered)
        field_centered = np.rot90(field_centered)
        bomb_field_centered = np.rot90(bomb_field_centered)
        opponent_field_centered = np.rot90(opponent_field_centered)
        explosion_field_centered = np.rot90(explosion_field_centered)
        best_coin_direction = best_coin_direction@np.array([[0,1,0,0], [0,0,1,0], [0,0,0,1], [1,0,0,0]])
        directional_feature = directional_feature@np.array([[0,1,0,0], [0,0,1,0], [0,0,0,1], [1,0,0,0]])
    

    res = np.concatenate((
                            coin_field_centered.flatten(),
                            field_centered.flatten(),
                            bomb_field_centered.flatten(),
                            opponent_field_centered.flatten(),
                            explosion_field_centered.flatten(),
                            np.array([bomb_available]),
                            np.array([bomb_effect]),
                            #best_coin_direction
                            directional_feature
                        ))

    #return best_coin_direction
    return res.reshape(-1)


### get valid actions - code adapted from rule_based_agent 
def reset_self(self):
    self.bomb_history = deque([], 5)


def get_valid_actions(self, game_state):
    # Checking, if the agent has to be updated 
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    # extracting the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    
    valid_tiles, valid_actions, action_ideas = [], [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    # Disallow the BOMB action if agent dropped a bomb in the same spot recently

    for (xb, yb), t in bombs:
        if (xb == x) and (abs(yb - y) < 4):
            
            # Run away
            if (yb > y): action_ideas.append('UP')
            if (yb < y): action_ideas.append('DOWN')
            # If possible, turn a corner
            action_ideas.append('LEFT')
            action_ideas.append('RIGHT')
        if (yb == y) and (abs(xb - x) < 4):
            # Run away
            if (xb > x): action_ideas.append('LEFT')
            if (xb < x): action_ideas.append('RIGHT')
            # If possible, turn a corner
            action_ideas.append('UP')
            action_ideas.append('DOWN')
    # Try random direction if directly on top of a bomb
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    if action_ideas: 
        action_list = [a for a in action_ideas if a in valid_actions]
        return(action_list)
    else: 
        return(valid_actions)
  