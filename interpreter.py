import copy
import numpy as np
import AI.Memory
import config as cf
#import scipy.stats as ss

def judge(history: AI.Memory.history):
    starting_state = history.events[1].state
    starting_state_hight_dist = get_col_distribution(starting_state, cf.CELL_NORM_VAL, float('inf'))
    ending_state = history.events[-1].state
    ending_state_hight_dist = get_col_distribution(ending_state, cf.CELL_NORM_VAL, 0)
    if max(starting_state_hight_dist) >= max(ending_state_hight_dist):
        #reward
        reward = cf.REWARD_RATE
        judge_events(history.events, 1, reward)
    else:
        #punisment
        punishment = cf.PENALTY_RATE * abs(max(ending_state_hight_dist) - max(starting_state_hight_dist))
        judge_events(history.events, -1, punishment)
    return history

def judge_events(events, mode, max_value):
    judgment = np.random.uniform(low=0, high=cf.NOISE_RATE, size=events[0].action[0].shape)
    if mode > 0:
        judgment = np.zeros(events[0].action[0].shape)
    for i in range(0, len(events)):
        rate = i / len(events)
        events[i].penalty = exp_interpolate(rate, cf.JUDGE_SPEED, max_value)
        judgment[0][events[i].action[1]] = mode * events[i].penalty# + events[i].score_delta - events[i].is_action_corr
        events[i].grade = judgment[0]
        #print("event ", i , " action: ", events[i].action[0], events[i].action[1], " grade: ", events[i].grade)

def exp_interpolate(x, speed, max_value):
    return max_value * x / (1 + speed * (1-x))

def get_row_distribution(field, thresh_value, min_value):
    result = []
    for i in range(0, cf.GAME_HIGHT):
        peek = 0
        for j in range(0, cf.GAME_WIDTH):
            if (field[i][j] == thresh_value or field[i][j] > min_value) and j > peek:
                peek = j
        result.append(peek)
    return result

def get_col_distribution(field, thresh_value, min_value):
    result = []
    for i in range(0, cf.GAME_WIDTH):
        peek = 0
        for j in reversed(range(0, cf.GAME_HIGHT)):
            h = cf.GAME_HIGHT - j
            if (field[j][i] == thresh_value or field[j][i] > min_value) and h > peek:
                peek = h
        result.append(peek)
    return result

def get_col_sum(field):
    col_sum = np.sum(field,axis=0)
    return col_sum

def get_row_sum(field):
    col_sum = np.sum(field,axis=1)
    return col_sum