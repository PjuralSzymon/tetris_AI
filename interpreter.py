import copy
import numpy as np
import scipy.stats as ss


def evaluate(game, old_figure, model_result, wrong_move_flag=1):
    action = np.argmax(model_result)
    result = model_result
    # is not correct
    if wrong_move_flag != 1:
        result[0][action] *= -1
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
        j = 0
        while True:
            if all(np.array(field[j]) == 0):
                j += 1
            else:
                break
        heights[i] = j
    best_action = np.argmax(heights)
    if best_action != action and heights[best_action] != heights[action]:
        rank = ss.rankdata(heights)
        result = result + rank
        result = result/np.linalg.norm(result)
    return result
