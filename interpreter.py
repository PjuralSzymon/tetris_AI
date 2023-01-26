import copy
import numpy as np
import scipy.stats as ss
from scipy.stats import skew
import config as cf
import statistics as stat

#
#   Funkcja evaluate zwraca result, importance
#   result czyli co POWINIEN zwrócić model dla danego wejścia
#   importance czyli jak ważne jest to żeby model się tego nauczył
#
def evaluate(game, old_figure, model_result, game_score, cf, wrong_move_flag=1):
    importance = 0
    action = np.argmax(model_result)
    result_random = np.random.uniform(low=0, high=cf.NOISE, size=model_result.shape)
    result = result_random
    col_sum_prev = get_col_sum(old_figure)
    col_sum_curr = get_col_sum(game.get_field_with_figure())
    col_dis_prev = get_col_distribution(old_figure)
    col_dis_curr = get_col_distribution(game.get_field_with_figure())

    max_height_prev = np.max(col_sum_prev)
    max_height_current = np.max(col_sum_curr)
    importance += abs(max_height_prev - max_height_current) * cf.importance_height_sum_max
    if max_height_prev < max_height_current:
        # max height is worst based on sums in columns
        result = punish(result, action, -cf.punishment * cf.importance_height_sum_max)
    elif  max_height_prev > max_height_current:
        # max height is better based on sums in columns
        result = punish(result, action, cf.punishment * cf.importance_height_sum_max)


    std_height_prev = stat.stdev(col_sum_prev)
    std_height_current = stat.stdev(col_sum_curr)
    importance += abs(std_height_prev - std_height_current) * cf.importance_height_sum_stdev
    if std_height_prev < std_height_current:
        # max height is worst based on sums in columns
        result = punish(result, action, -cf.punishment * cf.importance_height_sum_stdev)
    elif std_height_prev > std_height_current:
        # max height is better based on sums in columns
        result = punish(result, action, cf.punishment * cf.importance_height_sum_stdev)
        
    max_height_prev = np.max(col_dis_prev)
    max_height_current = np.max(col_dis_curr)
    importance += abs(max_height_prev - max_height_current) * cf.importance_height_dist_max
    if max_height_prev < max_height_current:
        # max height is worst based on absolute hight
        result[0][action] -= cf.punishment * cf.importance_height_dist_max
    elif max_height_prev > max_height_current:
        # max height is better based on absolute hight
        result[0][action] += cf.punishment * cf.importance_height_dist_max

    mean_height_prev = np.mean(col_dis_prev)
    mean_height_current = np.mean(col_dis_curr)
    importance += abs(mean_height_prev - mean_height_current) * cf.importance_height_dist_mean
    if mean_height_prev < mean_height_current:
        #importance -= 1
        result[0][action] -= cf.punishment * cf.importance_height_dist_mean
    elif mean_height_prev > mean_height_current:
        # max height is better based on absolute hight
        result[0][action] += cf.punishment * cf.importance_height_dist_mean

    result = threshold(result)
    return result, max(0, importance)

def punish(vector, action, punishment):
    for i in range(0, len(vector[0])):
        if i == action: vector[0][i] += punishment
        else: vector[0][i] -= punishment / 2
    return vector

def threshold(result):
    for i in range(0, len(result[0])):
        if result[0][i] < 0: result[0][i] = 0.0
        if result[0][i] > 1: result[0][1] = 1.0
    return result

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

def find_best_move(game, cf):
    def get_left(list): return sum(list[0:game.figure.x + 1])
    def get_right(list): return sum(list[game.figure.x + 1:len(list)])
    result = [0,0,0,0]
    sum_dist = get_col_sum(game.get_field_no_figure())
    if get_left(sum_dist) - get_right(sum_dist) < 1:
         result[1] += cf.NOISE
    elif get_left(sum_dist) - get_right(sum_dist) > 1:
        result[0] += cf.NOISE
    else:
        result[2] += cf.NOISE
    return result

def result_mean(r_1, r_2):
    result = []
    for i in range(0, len(r_1)):
        result.append((r_1[i] + r_2[i])/2)
    return result

def get_col_distribution(field, thresh_value=0.0):
    result = []
    for i in range(0, cf.GAME_WIDTH):
        peek = 0
        for j in reversed(range(0, cf.GAME_HIGHT)):
            h = cf.GAME_HIGHT - j
            if field[j][i] > thresh_value and h > peek:
                peek = h
        result.append(peek)
    return result