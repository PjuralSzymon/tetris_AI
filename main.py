import numpy as np
import AI.AI as AI
import interpreter
import sys
#from tetris import *
import time
from multiprocessing import Process
from multiprocessing import Pipe
from os import environ
import config as cf
import agent

if __name__ == '__main__':
    fps = 2
    size = (400, 500)
    epochs = 100
    gameplays = 5
    generation_size = 5
    processes = []
    pipe_connections = []
    name = "model_3.AI"
    best_model = AI.Model_RL(cf.GAME_WIDTH * cf.GAME_HIGHT, 4)
    best_score = -1
    model_id = 0    
    #best_model.summary()
    for e in range(0, epochs):
        results_in_epoch = []
        if best_model != None:
            conn1, conn2 = Pipe()
            pipe_connections.append(conn1)
            best_model_process = Process(target=agent.start_life, args=(conn2, size, fps, gameplays, best_score, e, model_id, best_model, False))
            processes.append(best_model_process)
        for i in range(0, generation_size):
            conn1, conn2 = Pipe()
            pipe_connections.append(conn1)
            new_model = best_model.deepcopy(min(i/(generation_size*2), 0.5))
            processes.append(Process(target=agent.start_life, args=(conn2, size, fps, gameplays, best_score, e, model_id, new_model, True)))
        for i in range(0, generation_size):
            processes[i].start()
        for i in range(0, generation_size):
            value = pipe_connections[i].recv()
            model = value[0]
            score = value[1]
            results_in_epoch.append(score)
            if score > best_score - 1:
                model_id += 1
                best_score = score
                best_model = model
        for i in range(0, generation_size):
            processes[i].join()
        best_model_acc = best_model.learn(1000)
        print("best_model acc: ", best_model_acc)
        processes.clear()
        pipe_connections.clear()
        print("epoch: ", e, " best_score: ", best_score, " score in epoch: ", np.round(results_in_epoch, 2))
        best_model.save(name)
    best_model.save(name)