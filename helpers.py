from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
import numpy as np
import random
import os

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


# def draw_punishment_graph(model_rl):
#     draw_smooth_graph(model_rl.punishment_hist, "Punishment hist", "smooth PH")
    # x_smooth = np.linspace(0, len(model_rl.punishment_hist), 50)
    # spl = make_interp_spline(range(len(model_rl.punishment_hist)),model_rl.punishment_hist,  k=3)
    # y_smooth = spl(x_smooth)
    
    # plt.plot(model_rl.punishment_hist, color='gray', linestyle='--', label='punishment history for best model')
    # plt.plot(x_smooth, y_smooth, '-', color='blue', label='Gładkie przybliżenie')
    # plt.xlabel('epochs')
    # plt.ylabel('punishment_hist')
    # plt.title('Wykres kary')
    # plt.legend()
    # plt.show()


def save_txt(path, decription):
    with open(path+'/decription.txt', 'w') as f:
        f.write(decription)
    f.close()

def draw_smooth_graph(data, label_norm, label_smooth, path, save_mode = False):
    if len(data) < 1000:
        draw_norm_graph(data, label_norm, path, save_mode)
        return
    x_smooth = np.linspace(0, len(data), 50)
    spl = make_interp_spline(range(len(data)),data,  k=3)
    y_smooth = spl(x_smooth)
    
    plt.plot(data, color='gray', linestyle='--', label=label_norm)
    plt.plot(x_smooth, y_smooth, '-', color='blue', label=label_smooth)
    plt.xlabel('epochs')
    plt.ylabel(label_norm)
    plt.title(label_norm)
    plt.legend()
    if save_mode:
        if os.path.isfile(path+"/"+label_norm+'.png'):
            os.remove(path+"/"+label_norm+'.png') 
        plt.savefig(path+"/"+label_norm+'.png')
    else:
        plt.show()
    plt.close()


def draw_norm_graph(data, label_norm,path, save_mode = False):
    plt.plot(data, color='gray', linestyle='--', label=label_norm)
    plt.xlabel('epochs')
    plt.ylabel(label_norm)
    plt.title(label_norm)
    plt.legend()
    if save_mode:
        if os.path.isfile(path+"/"+label_norm+'.png'):
            os.remove(path+"/"+label_norm+'.png') 
        plt.savefig(path+"/"+label_norm+'.png')    
    else:
        plt.show()
    plt.close()
