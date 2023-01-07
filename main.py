import numpy as np
import AI.AI as AI
import interpreter
import sys
from tetris import *
import time
from multiprocessing import Process
from multiprocessing import Pipe
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import config as cf



def start_game(pipe_conn, size, fps, n_gameplays, model_score = 0, epoch = 0, model_id = 0, _model_RL = None, hidden_mode = True, fast_mode = True):
    if hidden_mode == False:
        pygame.init()
        screen = pygame.display.set_mode(size)
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

        if counter % fps == 0:
            if game.state == "start":
                game.go_down()

        # Make decision by model:
        #figure_save_no_fig = game.get_field_no_figure()
        figure_save = game.get_field_with_figure()
        model_result, action = model_RL.move(figure_save)
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
        grade, importance = interpreter.evaluate(game, figure_save, np.round(model_result,2), game.score, correct_move_flag)
        if importance > 0:
            model_RL.grade(game.field, grade, importance)
        #print("model_result: "+ str(model_result) + " grade: "+ str(grade))
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
            text_game_over = font1.render("Game Over", True, (255, 125, 0))
            screen.blit(text1, [0, 0])
            screen.blit(text2, [0, 15])
            screen.blit(text3, [0, 30])
            screen.blit(text4, [0, 45])
            screen.blit(text5, [0, 60])
        if game.state == "gameover":
            if hidden_mode == False:
                screen.blit(text_game_over, [20, 200])
            model_score += interpreter.end_state_grade(game)
            game.reset()
            n_gameplays -= 1
        if hidden_mode == False: pygame.display.flip()
        if fast_mode == False and hidden_mode == False: clock.tick(fps)
    if hidden_mode == False: pygame.quit()
    pipe_conn.send([model_RL, model_score])
    return 1


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
    epochs = 1000
    gameplays = 30
    generation_size = 14
    processes = []
    pipe_connections = []
    name = "model_3.AI"
    global_best_model = AI.Model_RL(cf.GAME_WIDTH * cf.GAME_HIGHT, 4)
    local_best_model = AI.Model_RL(cf.GAME_WIDTH * cf.GAME_HIGHT, 4)
    global_best_score = 0
    model_id = 0    
    #best_model.summary()
    for e in range(0, epochs):
        results_in_epoch = []
        # always add global best model
        conn1, conn2 = Pipe()
        pipe_connections.append(conn1)
        best_model_process = Process(target=start_game, args=(conn2, size, fps, gameplays, global_best_model, e, model_id, global_best_model, False))
        processes.append(best_model_process)
        for i in range(0, generation_size):
            # reproduce new best model based on best model from last epoch
            conn1, conn2 = Pipe()
            pipe_connections.append(conn1)
            new_model = local_best_model.deepcopy(min(i/(generation_size*2), 0.1))
            processes.append(Process(target=start_game, args=(conn2, size, fps, gameplays, local_best_model, e, model_id, new_model, True)))
        for i in range(0, generation_size):
            processes[i].start()
        # find local best model ( best model from last epoch )
        local_best_score = 0
        for i in range(0, generation_size):
            value = pipe_connections[i].recv()
            model = value[0]
            score = value[1]
            results_in_epoch.append(score)
            if score > local_best_score:
                model_id += 1
                local_best_score = score
                local_best_model = model
        # if model from last epoch (local) is better than global replace
        if local_best_score > global_best_score:
            global_best_score = local_best_score
            global_best_model = local_best_model
        for i in range(0, generation_size):
            processes[i].join()
        processes.clear()
        pipe_connections.clear()
        print("epoch: ", e, " global_best_score: ", global_best_score, " local_best_score: ", local_best_score, " score in epoch: ", np.round(results_in_epoch, 2))
        global_best_model.save(name)
    global_best_model.save(name)