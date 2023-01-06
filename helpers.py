import numpy as np
import random

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