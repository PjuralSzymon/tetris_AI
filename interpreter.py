import copy
import numpy as np
import scipy.stats as ss

def evaluate(game, old_figure, model_result, game_score, wrong_move_flag=1):
    noise = 0.2
    #max_penalty = -1.0
    #max_award = 1.0
    importance = 1
    action = np.argmax(model_result)
    #result = np.random.rand(3,2) #np.ones(model_result.shape)
    result = np.random.uniform(low=0, high=noise, size=model_result.shape)
    #print("model_result: ", model_result)
    #print("result; ", result)
    max_height_prev = np.sum(get_col_sum(old_figure))
    max_height_current = np.sum(get_col_sum(game.get_field_with_figure()))
    if max_height_prev < max_height_current:
        # max height is worst
        #print("max h is worst")
        result[0][action] = 0.0
    else:
        # max height is better
        importance = 5
        result[0][action] = 0.9
        #print("max h is better")

    # if mean height cahnged
    mean_height_prev = np.mean(get_col_sum(old_figure))
    mean_height_current = np.mean(get_col_sum(game.get_field_with_figure()))
    if mean_height_prev < mean_height_current:
        # mean height is worst
        #print("mean is worst")
        result[0][action] = 0.0
    else:
        # mean height is better
        importance = 3
        #print("mean is better")
        result[0][action]  = 0.8
    if wrong_move_flag != 1:
        importance = 5
        result[0][action] = 0
    result[0][action] += game_score
    #print("final: ", result)
    return result, importance

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

def get_col_sum(field):
    field[field > 0] = 1.0
    col_sum = np.sum(field,axis=0)
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
