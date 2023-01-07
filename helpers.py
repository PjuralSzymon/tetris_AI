import numpy as np
import random
import config as cf

def create_model_input(field, extra_dim):
    input = np.array(field).flatten()
    if extra_dim:
        return input.reshape((cf.GAME_WIDTH * cf.GAME_HIGHT, 1))
    return input.reshape((cf.GAME_WIDTH * cf.GAME_HIGHT))

def binary_list(input_list, high_val, low_val):
    output_list = []
    for sublist in input_list:
        new_sublist = []
        for value in sublist:
            if value > 0:
                new_sublist.append(high_val)
            else:
                new_sublist.append(low_val)
        output_list.append(new_sublist)
    return output_list

def numpy2str(M):
    x = M.shape[0]
    y = M.shape[1]
    result = ""
    for i in range(0,x):
        for j in range(0,y):
            result += str(M[i][j])
            result += ','
    result = result[:-1]
    return result

def str2numpy(txt, x, y):
    split = txt.split(',')
    result = np.zeros((x, y))
    counter = 0
    for i in range(0,x):
        for j in range(0,y):
            result[i][j] = split[counter]
            counter += 1
    return result

def randomize_matrix(M, diffrence_rate):
    x = M.shape[0]
    y = M.shape[1]
    _min = np.min(M)
    _max = np.max(M)
    #print(_min, _max)
    if diffrence_rate < 0 or diffrence_rate > 1.0: print("diffrence_rate should belong to (0,1)")
    result = np.copy(M)
    for i in range(0,x):
        for j in range(0,y):
            diff =  ((1- diffrence_rate) * M[i][j] + diffrence_rate * random.uniform(_min, _max))/2
            result[i][j] = diff
    return result