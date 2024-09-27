import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .model import DQN
from collections import namedtuple, deque

def check_direction(coin_card,card, selfx, selfy, direction):
    """ 
    Checking if a given direction is valid and leads to a coin
    : param coin_card: array representing the game field with marked positions of coins
    : param card: array representing the current game field
    : param selfx: x-coordinate of the agent's position  
    : param selfy: y-coordinate of the agent's position  
    : param direction: direction that is going to be checked

    ----return----
    : valid: boolean, True if agent can move in direction
    : coin: boolean, True if there is a coin on (new_x, new_y)
    : new_x, new_y: new position after moving in direction 
    """
    DIRECTIONS = [ "UP", "RIGHT", "DOWN", "LEFT" ]
    COORDINATES = [(0,-1), (1,0), (0,1), (-1,0)]
    valid = False
    coin = False
    field_size_x, field_size_y = card.shape
    delta_x, delta_y = COORDINATES[DIRECTIONS.index(direction)]
    new_x = selfx+delta_x
    new_y = selfy+delta_y
    # checking if new positions are inside the field or not
    if new_x<0 or new_x >= field_size_x or new_y<0 or new_y >= field_size_y:
        return valid, coin, new_x, new_y
    # checking if movement is valid
    if card[new_x, new_y] != 0: 
        return valid, coin, new_x, new_y
    valid = True
    # checking if there is a coin on the new position
    if coin_card[new_x,new_y] != 0: 
        coin = True 
    return valid, coin, new_x, new_y
    

def best_coin_direction(coin_card, card, pos_self): 
    """ 
    searching for the first move on the shortest path to the nearest coin using 
    a breadth-first search  
    : param coin_card: array representing the game field with marked positions of coins
    : param card: array representing the current game field
    : param pos_self: coordinate tuple of the agent's position
    ----return----
    0, if no coin is found, otherwise first decision
    """
    iteration = 0
    DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]
    selfx, selfy = pos_self
    position = namedtuple('position',('posx', 'posy', 'first_decision'))
    unfinished_fields = deque()
    for direction in DIRECTIONS: 
        valid, coin, x, y = check_direction(coin_card, card, selfx, selfy, direction)
        if coin: 
            return direction
        if valid: 
            unfinished_fields.append(position(x,y,direction))
    if np.sum(coin_card) == 0: 
        return 0
    coin_found = False
    while not coin_found and iteration<10000: 
        current_field = unfinished_fields.popleft()
        for direction in DIRECTIONS: 
            valid, coin, x, y = check_direction(coin_card, card, current_field.posx, current_field.posy, direction)
            
            if coin: 
                coin_found = coin
                return current_field.first_decision
            if valid: 
                unfinished_fields.append(position( x,y,current_field.first_decision))
        iteration +=1
    return 0


def coin_feature(coin_card, card, pos_self):
    """ 
    output of best_coin_direction is shaped into a 4d array (one-hot)
    """
    DIRECTIONS = ["UP", "RIGHT", "DOWN", "LEFT"]
    res = np.zeros(4)
    best_dir = best_coin_direction(coin_card,card,pos_self)
    if best_dir != 0:
        res[DIRECTIONS.index(best_dir)]=1 
    return res


def mortality(bombs, card): 
    """ 
    produces a card containing all bombs and all fields in the danger zone of a bomb, danger zones depending 
    on the time till explosion
    : param bombs: list of bombs provided by game_state
    : param card: array representing the current game field
    """
    l = card.shape[0]
    # template of the danger zone 
    danger_zone = np.zeros((9,9))
    danger_zone[4] = np.array([1,5,10,20,30,20,10,5,1])
    danger_zone[:,4] = np.array([1,5,10,20,30,20,10,5,1])
    bomb_card_extended = np.zeros((l+8, l+8))
    # marking bombs on the card and also fields that will be effected by bomb
    for bomb in bombs: 
        (x,y),T = bomb 
        mask = np.zeros((9,9))
        for s in [-1,1]:
            for t in [0,1]:
                for step in np.arange(3):
                    posx = x+s*(1-t)*step
                    posy = y+s*t*step 
                    if 0<=posx<l and 0<=posy<l:
                        if card[posx,posy] >= 0: 
                            mask[posx-x+4,posy-y+4] = 1
                        else:
                            break
        bomb_card_extended[x:x+9,y:y+9] = bomb_card_extended[x:x+9,y: y+9]+ (3-T)*danger_zone*mask
    bomb_card = bomb_card_extended[4:4+l, 4:4+l]
    return bomb_card
         
def bomb_efficiency(card, posx, posy): 
    """ 
    returns the total damage caused by dropping a bomb at (posx, posy)
    : param card: array representing the current game field
    : param posx, posy: coordinates
    """
    l = card.shape[0]
    effect =0
    for s in [-1,1]:
        for t in [0,1]:
            for step in np.arange(3):
                x = posx+s*(1-t)*step
                y = posy+s*t*step 
                if 0<=x<=l-1 and 0<=y<=l-1:
                    if card[x,y] > 0: 
                        effect = effect + card[x,y] 
                    else:
                        break
    return effect 

def calculate_directional_feature(coin_field, field):
    """ 
    produces 4d-array with some weighted sums of the coins collactable if you move 
    in one of the possible directions
    : param coin_field: array representing the game field with marked positions of coins
    : param field: array representing the current game field
    """
    directional_feature = np.zeros((4,1))
    crate_field = field
    crate_field[field<0]=0
    if field[7,8] <0 and field[9,8] < 0:
        for i in np.arange(7): 
            directional_feature[2] = directional_feature[2] + 1/(i+1)* (np.sum(coin_field[:,8+i])+ np.sum(crate_field[:,8+i]))
            directional_feature[0] = directional_feature[0] + 1/(i+1)* (np.sum(coin_field[:,8-i])+ np.sum(crate_field[:,8-i]))
    elif field[8,7] <0 and field[8,9] < 0:
        for i in np.arange(7): 
            directional_feature[1] = directional_feature[1] + 1/(i+1)* (np.sum(coin_field[8+i,:])+ np.sum(crate_field[8+i,:]))
            directional_feature[3] = directional_feature[3] + 1/(i+1)* (np.sum(coin_field[8-i,:])+  np.sum(crate_field[8-i,:]))
    else: 
        for i in np.arange(7): 
            directional_feature[1] = directional_feature[1] + 1/(i+1)* (np.sum(coin_field[8+i,8-i:8+i]) + np.sum(crate_field[8+i,8-i:8+i]))
            directional_feature[2] = directional_feature[2] + 1/(i+1)* (np.sum(coin_field[8-i:8+i,8+i]) + np.sum(crate_field[8-i:8+i,8+i]))
            directional_feature[0] = directional_feature[0] + 1/(i+1)* (np.sum(coin_field[8-i:8+i,8-i]) + np.sum(crate_field[8-i:8+i,8-i]))
            directional_feature[3] = directional_feature[3] + 1/(i+1)* (np.sum(coin_field[8-i,8-i:8+i]) + np.sum(crate_field[8-i,8-i:8+i]))
    return directional_feature.flatten()