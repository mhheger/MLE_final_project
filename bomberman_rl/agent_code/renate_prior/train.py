from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random 
import pickle
import matplotlib.pyplot as plt
import csv
import os

from typing import List

import events as e
from .callbacks import state_to_features, SIZE

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
rng = np.random.default_rng()



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'dones'))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1500  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
BATCH_SIZE = 1024
GAMMA = 0.9
LR = 1e-5
TAU = 0.05
TARGET_UPDATE = 10
ROTATE = False
SAVE_STATS = True
# Events
DECREASE_COIN_DISTANCE = 'DECREASE_COIN_DISTANCE'
INCREASE_COIN_DISTANCE  = 'INCREASE_COIN_DISTANCE'
CORRECT_DIRECTION = 'CORRECT_DIRECTION'
INCORRECT_DIRECTION = 'INCORRECT_DIRECTION'
INCREASE_BOMB_EFFECT = 'INCREASE_BOMB_EFFECT'
DECREASE_BOMB_EFFECT = 'DECREASE_BOMB_EFFECT'
EXIT_DANGER_ZONE = 'EXIT_DANGER_ZONE'
ENTER_DANGER_ZONE = 'ENTER_DANGER_ZONE'
INCREASE_RISK = 'INCREASE_RISK'
DECREASE_RISK = 'DECREASE_RISK'
CRATE_WILL_BE_DESTROYED = 'CRATE_WILL_BE_DESTROYED'
WILL_DIE = 'WILL_DIE'
ENTER_EXPLOSION_AREA = 'ENTER_EXPLOSION_AREA'


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.loss = []
    self.model.policy_net.to(device)
    self.model.target_net.to(device)
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Calculate coin distances before and after the action
    distances=[[],[]]
    (agent_posx_old, agent_posy_old) = old_game_state["self"][3]
    (agent_posx_new, agent_posy_new) = new_game_state["self"][3]
    for (cx,cy) in old_game_state["coins"]:
        distance_old = abs(cx-agent_posx_old)+ abs(cy-agent_posy_old)
        if cx == agent_posx_old :
            if cx < 14:
                if old_game_state["field"][cx + 1, cy] == -1:
                    distance_old = distance_old + 2
            else :
                if old_game_state["field"][cx - 1, cy] == -1:
                    distance_old = distance_old + 2
        elif cy == agent_posy_old :
            if cy < 14:
                if old_game_state["field"][cx , cy + 1] == -1:
                    distance_old = distance_old + 2
            else :
                if old_game_state["field"][cx , cy -1] == -1:
                    distance_old = distance_old + 2

        distances[0].append(distance_old)
    for (cx,cy) in new_game_state["coins"]:
        distance_new = abs(cx-agent_posx_new) + abs(cy-agent_posy_new)
        if cx == agent_posx_old :
            if cx < 14:
                if new_game_state["field"][cx + 1, cy] == -1:
                    distance_new = distance_new + 2
            else :
                if new_game_state["field"][cx , cy + 1] == -1:
                    distance_new = distance_new + 2
        elif cy == agent_posy_old :
            if cy < 14:
                if new_game_state["field"][cx , cy + 1] == -1:
                    distance_new = distance_new + 2
            else :
                if new_game_state["field"][cx , cy - 1] == -1:
                    distance_new = distance_new + 2
        distances[1].append(distance_new)
    if len(distances[0]) != 0 and len(distances[1])!=0:
        minimal_distance_old = np.min(distances[0])
        minimal_distance_new = np.min(distances[1])
        if minimal_distance_new < minimal_distance_old or len(distances[0])< len(distances[1]):
            events.append(DECREASE_COIN_DISTANCE)
        else: 
            events.append(INCREASE_COIN_DISTANCE)
    
    # Reward correct directional movement
    features_old = state_to_features(old_game_state)
    features_new = state_to_features(new_game_state)
    directional_feature = features_old[-4:]
    best_direction = ACTIONS[np.argmax(directional_feature)]

    if self_action in ACTIONS[:4]:  # Check if the action was a directional move
        if best_direction == self_action:
            events.append(CORRECT_DIRECTION)
        else:
            events.append(INCORRECT_DIRECTION)

    # Reward increasements of bomb-effects
    bomb_effect_old = features_old[-5]
    bomb_effect_new = features_new[-5]
    if bomb_effect_new > bomb_effect_old: 
        events.append(INCREASE_BOMB_EFFECT)
    elif bomb_effect_new < bomb_effect_old:  
        events.append(DECREASE_BOMB_EFFECT)
    
    # Rewards for minimizing risk 
    field_size = SIZE*SIZE
    card = features_old[field_size:(2*field_size)].reshape((SIZE,SIZE))
    bomb_card_old = features_old[(2*field_size):(3*field_size)]
    bomb_card_new = features_new[(2*field_size):(3*field_size)]
    explosion_card = features_new[(3*field_size):(4*field_size)]
    index_self_pos = int(np.floor(field_size/2))
    risk_old = bomb_card_old[index_self_pos]
    risk_new = bomb_card_new[index_self_pos]
    explosion_state = explosion_card[index_self_pos]

    if risk_old >= risk_new: 
        events.append(DECREASE_RISK)
        if risk_old != 0 and risk_new == 0: 
            events.append(EXIT_DANGER_ZONE)
    elif risk_old < risk_new: 
        events.append(INCREASE_RISK)
        if risk_old == 0: 
            events.append(ENTER_DANGER_ZONE)
    if explosion_state != 0: 
        events.append(ENTER_EXPLOSION_AREA)
    
    if self_action == 'BOMB' and bomb_effect_old !=0:
        events.append('CRATE_WILL_BE_DESTROYED')
    if self_action == 'BOMB': 
        if no_escape(card):
            #print("You will die")
            events.append(WILL_DIE)
    
    # Store transitions with data augmentation via rotation
    if ROTATE:
        for i in np.arange(4):
            rotated_state_old = state_to_features(old_game_state, rotate=i)
            rotated_state_new = state_to_features(new_game_state, rotate=i)
            rotated_action = rotate_action(self_action, rotate=i)
            reward = reward_from_events(self, events)
            self.transitions.append(Transition(rotated_state_old, rotated_action, rotated_state_new, reward, 0))
    else: 
        state_old = state_to_features(old_game_state)
        state_new = state_to_features(new_game_state)
        reward = reward_from_events(self, events)
        action = self_action
        self.transitions.append(Transition(state_old, action, state_new, reward, 0))
    if SAVE_STATS:
        update_stats(self, events, reward)

    optimize_model(self, self.transitions)
    # Optimize model periodically 
    
    
   

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    if ROTATE:
        for i in np.arange(4):
            rotated_state_last = state_to_features(last_game_state, rotate=i)
            rotated_action = rotate_action(last_action, rotate=i)
            reward = reward_from_events(self, events)
            self.transitions.append(Transition(rotated_state_last, rotated_action, rotated_state_last, reward, 1))
    else: 
        self.transitions.append(Transition(state_to_features(last_game_state), last_action, state_to_features(last_game_state), reward_from_events(self, events), 1))
    optimize_model(self, self.transitions)
    # Store the model
    self.model.epochs[0] = self.model.epochs[0] + 1

    self.stats["mean_loss"] = np.mean(self.loss)
    self.loss = []
    self.stats["score"] = last_game_state["self"][1] 
    if SAVE_STATS:
        if not os.path.isfile("stats.csv"):
            with open('stats.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["steps_survived", "coins_collected", "crates_destroyed", "opponents_killed", "invalid_actions","score" ,"total_reward", "mean_loss"])  # Writing headers
        
        with open('stats.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(list(self.stats.values()))
            reset_stats(self)

    with open("renate-model.pt", "wb") as file:
        self.model.policy_net.cpu()
        self.model.target_net.cpu()
        pickle.dump(self.model, file)
        self.model.policy_net.to(device)
        self.model.target_net.to(device)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: -30,
        e.MOVED_DOWN: -2, 
        e.MOVED_LEFT: -2, 
        e.MOVED_RIGHT: -2,
        e.MOVED_UP: -2,
        e.COIN_COLLECTED: 60,
        e.KILLED_OPPONENT: 10,
        e.COIN_FOUND: 60,
        e.CRATE_DESTROYED: 20, 
        e.KILLED_SELF: -80, 
        e.GOT_KILLED: -10, 
        e.OPPONENT_ELIMINATED: 10, 
        e.SURVIVED_ROUND: 10,
        e.INVALID_ACTION: -30 , 
        e.BOMB_DROPPED: 2,
        DECREASE_COIN_DISTANCE: 16,
        INCREASE_COIN_DISTANCE: -15, 
        INCREASE_BOMB_EFFECT: 1,
        DECREASE_BOMB_EFFECT: -15,
        EXIT_DANGER_ZONE : 20,
        ENTER_DANGER_ZONE : -15,
        INCREASE_RISK : -3,
        DECREASE_RISK : 2,
        CRATE_WILL_BE_DESTROYED: 20,
        #WILL_DIE: -60,  
        ENTER_EXPLOSION_AREA: -20,
        CORRECT_DIRECTION: 4,
        INCORRECT_DIRECTION: -3
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def optimize_model(self, transition_batch):
    optimizer = optim.Adam(self.model.policy_net.parameters(), lr=LR)
    #if len(transition_batch) < BATCH_SIZE:
         #return
    #batch = random.sample(transition_batch, BATCH_SIZE)
    
    states, actions, next_states, rewards, dones = zip(*transition_batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor([ACTIONS.index(action) for action in actions]).to(device)
    rewards = torch.FloatTensor(np.array(rewards)).to(device)
    
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor([float(done) for done in dones]).to(device)

    current_q_values = self.model.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = self.model.target_net(next_states).max(1)[0]
    target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

    indices = random.choices(np.arange(len(transition_batch)), weights= F.softmax(target_q_values, dim =0), k = BATCH_SIZE)

    loss = nn.SmoothL1Loss()(current_q_values[indices], target_q_values.detach()[indices])
    self.loss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

   
    if self.model.epochs[0] % TARGET_UPDATE == 0: 
        self.model.target_net.load_state_dict(self.model.policy_net.state_dict()) 



def rotate_action(action, rotate=0):
    ACTIONS_ROTATED = ['RIGHT', 'DOWN', 'LEFT', 'UP', 'WAIT', 'BOMB']
    index = ACTIONS.index(action)
    for i in np.arange(rotate):
        index = ACTIONS.index(ACTIONS_ROTATED[index])
    return ACTIONS[index]



def no_escape(card):
    #checking escape routes

    up = int(np.ceil(SIZE))
    low = SIZE - up
    for i in [0,1]:
        for j in [-1,1]:
            for k in np.arange(1,up):
                delta_x = (1-i)*j*k
                delta_y = i*j*k 
                x = low + delta_x
                y = low + delta_y
                if card[x,y] == 0: 
                    for l in [-1,1]: 
                        x_check = x + i*l 
                        y_check = y + (1-i)*l
                        if 0<= x_check < SIZE and 0 <= y_check < SIZE: 
                            if card[x_check,y_check] == 0.0 :
                                return False
                else:
                    break 
    return True

def update_stats(self, events, reward): 
    if e.COIN_COLLECTED in events: 
        self.stats["coins_collected"]+=1
    if e.KILLED_OPPONENT in events: 
        self.stats["opponents_killed"]+=1
    if not e.KILLED_SELF in events:
        if not (e.GOT_KILLED in events):
            self.stats["steps_survived"] += 1
    if e.INVALID_ACTION in events: 
        self.stats["invalid_actions"] += 1
    self.stats["crates_destroyed"] += events.count(e.CRATE_DESTROYED)
    self.stats["total_reward"] += reward

def reset_stats(self): 
    self.stats = {
        "steps_survived": 0,
        "coins_collected": 0, 
        "crates_destroyed": 0, 
        "opponents_killed": 0, 
        "invalid_actions": 0,
        "score":0,
        "total_reward": 0,
        "mean_loss": 0
    }
    self.pos_history = deque(maxlen = 20 )

def get_lr(rounds):
    if rounds < 25000:
        return 1e-4
    return 1e-5
