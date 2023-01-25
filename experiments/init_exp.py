from multiprocessing import set_start_method
from multiprocessing import Process
from multiprocessing import Pipe
import matplotlib.pyplot as plt
import sys
import os
import msvcrt
import time
sys.path.append("../")
import interpreter
import main
import config
import AI.AI as AI
import random




if __name__ == '__main__':

    id = 0
    id_add_on = 'P4_'
    config_save = {}
    gameplays = 500
    generation_size = 16
    precission = 4
    pipe_connections = []
    processes = []
    results = []
    
    path = "../results/init_experiment_2/"
    save_txt_name = id_add_on + "result_history.txt"
    save_graph_name = id_add_on +"best_results.png"
    with open(path + save_txt_name, "w") as file:
        while True:
            print("calucaltion for epoch: 1, ended write Q to quit if not algorithm will continue authomatically")
            if msvcrt.kbhit():
                input_str = msvcrt.getche()
                print("You entered:", input_str)
                if 'q' in str(input_str):
                    break
            else:
                time.sleep(3)
                
            #zmienne do przebadania:
            
            #background_process.start()
            for i in range(0, generation_size):
                id += 1
                conn1, conn2 = Pipe()
                pipe_connections.append(conn1)
                config.NOISE = round(random.uniform(0.8, 0.1), precission)
                config.NOISE_END_VAL = round(config.NOISE / float((random.randint(10,1000))), precission)
                config.IMPORTANCE_THRESH = round(random.uniform(0.1, 0.8), precission)
                config.punishment = round(random.uniform(0.1, 2.0), precission)
                config.importance_height_sum_max = round(random.uniform(0.1, 0.8), precission)
                config.importance_height_sum_stdev = round(random.uniform(0.1, 0.8), precission)
                config.importance_height_dist_max = round(random.uniform(0.1, 0.8), precission)
                config.importance_height_dist_mean = round(random.uniform(0.1, 0.8), precission)
                config.learning_rate = round(random.uniform(0.001, 0.2), 2 * precission)
                label = "noise: "+str(config.NOISE) + " noise_ev: "+ str(config.NOISE_END_VAL) + " imp_thr: "+ str(config.IMPORTANCE_THRESH) + " punish: " + str(config.punishment) + "imp_h_sum_max: " + str(config.importance_height_sum_max) + " imp_h_sum_stdev: " + str(config.importance_height_sum_stdev) + " imp_h_dist_max: " + str(config.importance_height_dist_max) + " imp_h_dist_mean: " + str(config.importance_height_dist_mean) + " lr: " + str(config.learning_rate)
                config_save[str(id_add_on) + str(id)] = label
                print("id: ", str(id_add_on) + str(id))
                random_model = AI.Model_RL(config.GAME_WIDTH * config.GAME_HIGHT + 3, 4, False, str(id_add_on) + str(id))
                processes.append(Process(target=main.start_game, args=(conn2, gameplays, config.get_config_object(), 1, 1, random_model, True)))
            for i in range(0, generation_size):
                processes[i].start()
            
            for i in range(0, generation_size):
                value = pipe_connections[i].recv()
                model = value[0]
                score = value[1]
                file.write(f"{model.id}, {score} , {config_save[model.id]}\n")
                results.append([model.id, score])

            for i in range(0, generation_size):
                processes[i].join()
            processes.clear()
            pipe_connections.clear()
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:20]
        identifiers = [result[0] for result in top_results]
        values = [result[1] for result in top_results]
        plt.bar(identifiers, values)
        plt.xlabel("identifiers")
        plt.ylabel("values")
        #plt.show()
        plt.savefig(path+"/"+save_graph_name)
