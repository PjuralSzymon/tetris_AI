import copy
import numpy as np
import scipy.stats as ss


def evaluate(game, old_figure, model_result, wrong_move_flag=1):
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
    print(model_result)
    print(result)
    print('______________________')
    return result
