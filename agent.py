import numpy as np
import AI.AI as AI
import AI.Memory
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

def start_life(pipe_conn, size, fps, n_gameplays, model_score = 0, epoch = 0, model_id = 0, model_RL = None, hidden_mode = True, fast_mode = True):
    if hidden_mode == False:
        pygame.init()
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Tetris")
    if hidden_mode == False:clock = pygame.time.Clock()
    game = Tetris(cf.GAME_HIGHT, cf.GAME_WIDTH)
    # if _model_RL == None:
    #     model_RL = AI.Model_RL(game.width * game.height, 4)
    # else:
    #     model_RL = _model_RL
    counter = 0
    pressing_down = False
    model_score = 0.0
    last_history = AI.Memory.history()
    game.new_figure()
    acc = 0
    while n_gameplays > 0:
        counter += 1
        if counter > 100000:
            counter = 0

        if hidden_mode == False:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)

        # Make decision by model:
        figure_save = game.get_field_with_figure()
        model_result, action = model_RL.move(figure_save)
        new_figure_flag = 0
        correct_move_flag = 0
        score_delta = 0
        if action == 0:
            correct_move_flag = game.go_side(1)
        elif action == 1:
            correct_move_flag = game.go_side(-1)
        elif action == 2:
            correct_move_flag = game.rotate()
        else:
            #score_delta = game.go_down()
            pass

        if counter % fps == 0:
            if game.state == "start":
                score_delta, new_figure_flag = game.go_down()

        last_history.events.append(AI.Memory.event(figure_save, [model_result, action], score_delta, correct_move_flag))
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
        if new_figure_flag:
            #print("new_figure_flag, ", len(last_history.events))
            #memory.memories.append(last_history)
            interpreter.judge(last_history)
            model_RL.memory.add_history(interpreter.judge(last_history))
            acc = model_RL.learn()
            last_history = AI.Memory.history()
        if game.state == "gameover":
            if hidden_mode == False:
                screen.blit(text_game_over, [20, 200])
            game.reset()
            n_gameplays -= 1
        if hidden_mode == False: pygame.display.flip()
        if fast_mode == False and hidden_mode == False: clock.tick(fps)
    
    if hidden_mode == False: pygame.quit()
    print("acc: ", acc)
    pipe_conn.send([model_RL, game.score])
    return 1
