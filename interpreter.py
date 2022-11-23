import numpy as np

def evaluate(game, model_result, wrong_move_flag = 1):
    action = np.argmax(model_result)
    result = model_result
    if wrong_move_flag == 1:
        # is correct
        pass
    else:
        result[action-1] *= -1
    return result