import numpy as np
import AI.AI as AI
import interpreter
import sys
import time
from multiprocessing import Process
import matplotlib.pyplot as plt
from multiprocessing import Pipe
import os
from os import environ
from scipy.interpolate import make_interp_spline, BSpline
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from tetris import *

import pygame
import config as cf
import math
import helpers
import warnings
warnings.filterwarnings('ignore')

#conn2, size, fps, gameplays, new_model, e, model_id, new_model, True
def start_game(pipe_conn, n_gameplays, cf, epoch = 0, model_id = 0, _model_RL = None, hidden_mode = True, fast_mode = True):
    #print(os.getpid(), " config.noise: ", cf.NOISE, " modelid: ", _model_RL.id)
    model_score = 0
    if hidden_mode == False:
        pygame.init()
        screen = pygame.display.set_mode(cf.size)
        pygame.display.set_caption("Tetris")
    if hidden_mode == False:clock = pygame.time.Clock()
    game = Tetris(cf.GAME_HIGHT, cf.GAME_WIDTH)
    if _model_RL == None:
        model_RL = AI.Model_RL(game.width * game.height, 4)
    else:
        model_RL = _model_RL
    counter = 0
    pressing_down = False
    model_score = 0.0
    importance_hist = []
    
    #punishment_hist = []
    while n_gameplays > 0:
        if game.figure is None:
            game.new_figure()
        counter += 1
        if counter > 100000:
            counter = 0

        if hidden_mode == False:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

        if counter % cf.fps == 0:
            if game.state == "start":
                game.go_down()

        # Make decision by model:
        #figure_save_no_fig = game.get_field_no_figure()
        figure_save = game.get_field_with_figure()
        politics = [figure_save, game.figure.x + 1, game.figure.y, game.figure.type]
        model_result, action = model_RL.move(politics)
        correct_move_flag = 1
        
        if action == 0:
            correct_move_flag = game.go_side(1)
        elif action == 1:
            correct_move_flag = game.go_side(-1)
        elif action == 2:
            correct_move_flag = game.rotate()
        else:
            correct_move_flag = game.go_down()
        
        # grade made changes
        #print("action: "+ str(action))
        grade, importance = interpreter.evaluate(game, figure_save, np.round(model_result,2), game.score, cf, correct_move_flag)
        importance_hist.append(importance)
        if importance > cf.IMPORTANCE_THRESH:
            model_RL.grade(politics, grade, importance, cf.learning_rate)
        #punishment = sum(sum(model_result - grade))
        #model_RL.punishment_hist.append(punishment)
        #print("model_result: "+ str(model_result) + " grade: "+ str(grade)+ " punishment: " + str(punishment))
        #input(" ? ")
        if hidden_mode == False:
            screen.fill(cf.WHITE)
            
            for i in range(game.height):
                for j in range(game.width):
                    pygame.draw.rect(screen, cf.GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
                    if game.field[i][j] > 0:
                        pygame.draw.rect(screen, colors[game.field[i][j]],
                                        [game.x + game.zoom * j + 1, game.y + game.zoom * i + 1, game.zoom - 2, game.zoom - 1])

            if game.figure is not None:
                for i in range(4):
                    for j in range(4):
                        p = i * 4 + j
                        if p in game.figure.image():
                            pygame.draw.rect(screen, colors[game.figure.color],
                                            [game.x + game.zoom * (j + game.figure.x) + 1,
                                            game.y + game.zoom * (i + game.figure.y) + 1,
                                            game.zoom - 2, game.zoom - 2])

            font = pygame.font.SysFont('Calibri', 20, True, False)
            font1 = pygame.font.SysFont('Calibri', 65, True, False)
            text1 = font.render("Score: " + str(game.score), True, cf.BLACK)
            text2 = font.render("Gameplay: " + str(n_gameplays), True, cf.BLACK)
            text3 = font.render("Epoch: " + str(epoch), True, cf.BLACK)
            text4 = font.render("Model id: " + str(model_id), True, cf.BLACK) 
            text5 = font.render("Model score: " + str(round(model_score,2)), True, cf.BLACK) 
            text6 = font.render("importance: " + str(importance), True, cf.BLACK) 
            text7 = font.render("figurepos: " + str(game.figure.x) + ","+ str(game.figure.y), True, cf.BLACK) 

            text_game_over = font1.render("Game Over", True, (255, 125, 0))
            screen.blit(text1, [0, 0])
            screen.blit(text2, [0, 15])
            screen.blit(text3, [0, 30])
            screen.blit(text4, [0, 45])
            screen.blit(text5, [0, 60])
            screen.blit(text6, [0, 75])
            screen.blit(text7, [0, 90])

        if game.state == "gameover":
            if hidden_mode == False:
                screen.blit(text_game_over, [20, 200])
            model_score += interpreter.end_state_grade(game)
            game.reset()
            n_gameplays -= 1
        if hidden_mode == False: pygame.display.flip()
        if fast_mode == False and hidden_mode == False: clock.tick(cf.fps)
    if hidden_mode == False: pygame.quit()
    model_RL.last_game_score = model_score
    model_RL.importance_hist += importance_hist
    pipe_conn.send([model_RL, model_score])
    return 1


def save_graphs(punishment_hist, importance_hist, noise_hist, best_model_score_hist, path, description):
    helpers.draw_smooth_graph(punishment_hist, "Punishment_hist", "smooth_PH", path, True)
    helpers.draw_smooth_graph(importance_hist, "Importance_hist", "smooth_IH", path, True)
    helpers.draw_norm_graph(noise_hist, "noise_hist", path, True)
    helpers.draw_norm_graph(best_model_score_hist, "best_model_score_hist", path, True)
    helpers.save_txt(path, description)

# start = time.time()
# start_game(size, fps, 3,  _model_RL = None, fast_mode = True, hidden_mode = False)
# end = time.time()
# print("hidden_mode = False", end - start)
# start = time.time()
# start_game(size, fps, 3,  _model_RL = None, fast_mode = True, hidden_mode = True)
# end = time.time()
# print("hidden_mode = True", end - start)

if __name__ == '__main__':
    fps = 2
    size = (400, 500)
    epochs = 500
    gameplays = 100
    generation_size = 13
    processes = []
    pipe_connections = []
    model_name_save = "model_26"
    best_models = []
    noise_start_val = cf.NOISE
    noise_hist = []
    path = "./results/"+model_name_save
    description = "Ten model to proba odapalenia najlepszych konfiguracji znalezionych dla losowych modeli, przy uzyciu losowego startu, 100 rozgrywek i 14 agentow,\n\
    to samo co model_13 ale szum opiera sie na find best move i wyrzucilem z evaluate srednia a std jest dzielone na pol\n\
    model dodatkowo dostaje pozycje figury i jej typ przez co jest nie kompatybilny z innymi mdoelami\n\
    best_move na szumie nei jest liczony z polowy tylko z pozycji figury\n\
    Funkcje aktywacyjne są spowrotem sigmoid zostawiam ten model na 100 epok\n\
    Brak best move\n\
    punishment jest stosowane wdlug kary punishment a nie na pojedynczych indeksah\n\
    zmienilem tak ze kara i angroda ejst dodawana tylko jesli nastapila zmiana jesli nie to nic sie nie dzieje\n\
    Z powrotem dodalem ocene przez col distrybution \n\
    Sa to najlepsze ustawienia z init_experiment_2 \n\
    "
    if os.path.exists(path):
        print("Model with this name already exists")
        exit()
    os.mkdir(path, 0o666)
    for i in range(0, math.ceil(generation_size * 0.6)):
        best_models.append(AI.Model_RL(cf.GAME_WIDTH * cf.GAME_HIGHT + 3, 4))
        #best_models.append(AI.Model_RL.load("results\model_base.AI"))
    model_id = 0    
    punishment_hist = []
    importance_hist = []
    best_model_score_hist = []
    #   (pipe_conn, n_gameplays, cf, epoch = 0, model_id = 0, _model_RL = None, hidden_mode = True, fast_mode = True):
    #   processes.append(Process(target=main.start_game, args=(conn2, gameplays, config.get_config_object(), 1, 1, random_model, True)))
    for e in range(0, epochs):
        results_in_epoch = []
        conn1, conn2 = Pipe()
        pipe_connections.append(conn1)
        best_model_process = Process(target=start_game, args=(conn2, gameplays, cf.get_config_object(), e, model_id, best_models[0], False))
        processes.append(best_model_process)
        for i in range(0, generation_size):
            conn1, conn2 = Pipe()
            pipe_connections.append(conn1)
            if i < len(best_models):
                new_model = new_model = best_models[i].deepcopy(0.0)
            else:
                new_model = AI.Model_RL(cf.GAME_WIDTH * cf.GAME_HIGHT + 3, 4)
            processes.append(Process(target=start_game, args=(conn2, gameplays, cf.get_config_object(), e, model_id, new_model, True)))
        for i in range(0, generation_size):
            processes[i].start()
        # find local best model ( best model from last epoch )
        local_best_score = -1
        for i in range(0, generation_size):
            value = pipe_connections[i].recv()
            model = value[0]
            score = value[1]
            results_in_epoch.append(score)
            best_models.append(model)
        for i in range(0, generation_size):
            processes[i].join()
        best_models.sort(key=lambda x: x.last_game_score, reverse = True)
        best_models = best_models[0: math.ceil(generation_size * 0.6)]
        processes.clear()
        pipe_connections.clear()
        print("epoch: ", e, " best_models: ", [x.last_game_score for x in best_models], " score in epoch: ", results_in_epoch)
        #best_models[0].save(path+"/model.AI")
        #update noise level
        noise_hist.append(cf.NOISE)
        cf.NOISE = noise_start_val * (cf.NOISE_END_VAL/noise_start_val) ** (e / epochs)
        best_model_score_hist.append(best_models[0].last_game_score)
        punishment_hist += best_models[0].punishment_hist
        importance_hist += best_models[0].importance_hist
        save_graphs(punishment_hist, importance_hist, noise_hist, best_model_score_hist, path, description)
    best_models[0].save(path+"/model.AI")
    # draw_punishment_graph(best_models[0])
    # print("punishment_hist: ", type(punishment_hist))
    # print("punishment_hist: ", len(punishment_hist))
    # print("punishment_hist: ", punishment_hist)
    # print("importance_hist: ", type(importance_hist))
    # print("importance_hist: ", len(importance_hist))
    # print("importance_hist: ", importance_hist)
    save_graphs(punishment_hist, importance_hist, noise_hist, best_model_score_hist, path, description)
    print("DONE")