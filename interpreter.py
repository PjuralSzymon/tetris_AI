import copy
import numpy as np
import scipy.stats as ss
import config as cf

#
#   Funkcja evaluate zwraca result, importance
#   result czyli co POWINIEN zwrócić model dla danego wejścia
#   importance czyli jak ważne jest to żeby model się tego nauczył
#
def evaluate(game, old_figure, model_result, game_score, wrong_move_flag=1):
    importance = 0
    action = np.argmax(model_result)
    result = np.random.uniform(low=0, high=cf.NOISE, size=model_result.shape)
    max_height_prev = np.max(get_col_sum(old_figure))
    max_height_current = np.max(get_col_sum(game.get_field_with_figure()))
    if max_height_prev == max_height_current:
        # max height is better
        importance = 5
        result[0][action] = 0.5
        importance += 1
        #print("max h is the same")        
        pass
    elif max_height_prev < max_height_current:
        # max height is worst
        #print("max h is worst")
        result[0][action] = 0.0
        importance += 1
    else:
        # max height is better
        importance = 5
        result[0][action] = 0.5
        importance += 1
        #print("max h is better")
    peaks = get_highest_peaks(old_figure)
    # if mean height cahnged
    mean_height_prev = np.mean(get_col_sum(old_figure))
    mean_height_current = np.mean(get_col_sum(game.get_field_with_figure()))
    if mean_height_prev == mean_height_current:
        #importance -= 1
        pass
    elif mean_height_prev < mean_height_current:
        # mean height is worst
        #print("mean is worst")
        result[0][action] = 0.0
        importance += 1
    else:
        # mean height is better
        importance += 1
        #print("mean is better")
        result[0][action]  = 0.6
    if wrong_move_flag != 1:
        importance += 2
        result[0][action] = 0
    if game_score > 0:
        importance += 2
        result[0][action] += game_score
    #print("final: ", result)
    return result, max(0, importance)

def end_state_grade(game):
    row_dens_w = 0.0
    score_w = 1.0
    reward = 0.0
    row_states = get_row_sum(game.get_field_with_figure())
    row_dense_reward_sum = 0
    for r in row_states:
        row_dense_reward_sum += (r/game.width) * row_dens_w
    reward += row_dense_reward_sum
    reward += game.score * score_w
    return reward

def get_highest_peaks(field):
    field[field > 0] = 1.0
    field = field.transpose()
    x = field.shape[0]
    y = field.shape[1]
    result = []
    for i in range(0,x):
        max_j = 0
        for j in reversed(range(0,y)):
            if field[i][j] > 0:
                max_j = 20 - j
        result.append(max_j)
    return result

def get_col_sum(field):
    field[field > 0] = 1.0
    col_sum = np.sum(field,axis=0)
    return col_sum

def get_row_sum(field):
    field[field > 0] = 1.0
    col_sum = np.sum(field,axis=1)
    return col_sum

# def get_row_distribution(field, thresh_value=0.0):
#     result = []
#     for i in range(0, cf.GAME_HIGHT):
#         peek = 0
#         for j in range(0, cf.GAME_WIDTH):
#             if field[i][j] > thresh_value and j > peek:
#                 peek = j
#         result.append(peek)
#     return result

# def get_col_distribution(field, thresh_value=0.0):
#     result = []
#     for i in range(0, cf.GAME_WIDTH):
#         peek = 0
#         for j in reversed(range(0, cf.GAME_HIGHT)):
#             h = cf.GAME_HIGHT - j
#             if field[j][i] > thresh_value and h > peek:
#                 peek = h
#         result.append(peek)
#     return result