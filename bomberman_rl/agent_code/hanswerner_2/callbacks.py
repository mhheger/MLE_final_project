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
SIZE = 15
INPUT_SIZE = 5*SIZE*SIZE+1+1
P_START = 1.0 
P_END = 0.1
RATE = 0.0001
RATE2 = 1
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
        self.model = DeepQNet(DQN(SIZE, 6), DQN(SIZE, 6), [0])
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
    random_prob = P_END+ (P_START-P_END)*np.exp(-RATE*round*(1/(RATE2*game_state["round"]+1)))
    if self.train and random.random() < random_prob:
        #self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        choice = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        #choice = acting_rules(game_state)
        return(choice)

    self.logger.debug("Querying model for action.")
    #if not self.train and random.random() < P_END:
       # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    #print(self.model.policy_net.forward(state).detach().cpu().numpy())
    index =np.argmax(self.model.policy_net.forward(state).detach().cpu().numpy())
    #print(ACTIONS[index])
    return ACTIONS[index]

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
                            np.array([bomb_effect])
                            #best_coin_direction
                            #directional_feature
                        ))

    #return best_coin_direction
    return res.reshape(-1)