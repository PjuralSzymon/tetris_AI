import copy
import numpy as np
import scipy.stats as ss

def evaluate(game, old_figure, model_result, game_score, wrong_move_flag=1):
    noise = 0.4
    #max_penalty = -1.0
    #max_award = 1.0
    importance = 0
    action = np.argmax(model_result)
    #result = np.random.rand(3,2) #np.ones(model_result.shape)
    result = np.random.uniform(low=0, high=noise, size=model_result.shape)
    #print("model_result: ", model_result)
    #print("result; ", result)
    max_height_prev = np.max(get_col_sum(old_figure))
    max_height_current = np.max(get_col_sum(game.get_field_with_figure()))
    if max_height_prev == max_height_current:
        #importance -= 1
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
    # print("old_figure: ")
    # print(old_figure)
    # print("peaks: ")
    # print(peaks)
    # print("-----")
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


def evaluate3(game, old_figure, model_result, wrong_move_flag=1):
    max_penalty = -0.1
    max_award = 2.0
    action = np.argmax(model_result)
    result = np.ones(model_result.shape)
    # is not correct
    if wrong_move_flag != 1:
        result[0][action] = max_penalty
        return result
    # if max height cahnged
    max_height_prev = np.sum(get_col_sum(old_figure))
    max_height_current = np.sum(get_col_sum(game.get_field_with_figure()))
    if max_height_prev < max_height_current:
        # max height is worst
        result[0][action] = max_penalty/2
    else:
        # max height is better
        result = -np.ones(model_result.shape)
        result[0][action] = max_award/2
    # if mean height cahnged
    mean_height_prev = np.mean(get_col_sum(old_figure))
    mean_height_current = np.mean(get_col_sum(game.get_field_with_figure()))
    if mean_height_prev < mean_height_current:
        # mean height is worst
        result[0][action] = max_penalty/3
    else:
        # mean height is better
        result = -np.ones(model_result.shape)
        result[0][action] = max_award/3
    return np.multiply(result,model_result)

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

def evaluate2(game, old_figure, model_result, wrong_move_flag=1):
    action = np.argmax(model_result)
    result = model_result
    # is not correct
    if wrong_move_flag != 1:
        result[0][action] = 0
        return result
    # is correct
    heights = [None]*4
    for i in range(len(result[0])):
        correct_move_flag = 1
        game_copy = copy.deepcopy(game)
        game_copy.figure = old_figure
        if action == 1:
            correct_move_flag = game_copy.rotate()
        elif action == 2:
            correct_move_flag = game_copy.go_side(-1)
        elif action == 3:
            correct_move_flag = game_copy.go_side(1)
        if correct_move_flag != 1:
            heights[i] = -1
        game_copy.go_down()
        field = game_copy.get_field_with_figure()
        heights_of_tower = [0]*len(field[0])
        for k in range(len(field[0])):
            for j in range(len(field)):
                if field[j][k] == 0:
                    heights_of_tower[k] += 1
        heights[i] = sum(heights_of_tower)
    if heights[np.argmax(heights)] != heights[action]:
        rank = ss.rankdata(heights)
        result[0] = result[0] + rank
        result = result/np.linalg.norm(result)
    else:
        result[0][action] *= 1.1
        if result[0][action] > 1:
            result[0][action] = 1
    #print(model_result)
    #print(result)
    #print('______________________')
    return result
