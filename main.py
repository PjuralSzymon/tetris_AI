import pygame
import numpy as np
import AI.AI as AI
import interpreter
import sys
from tetris import *
import time
from multiprocessing import Process


# Initialize the game engine
pygame.init()

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

def start_game(size, fps, n_gameplays,  _model_RL = None, fast_mode = True, hidden_mode = False):
    if hidden_mode == False:
        screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Tetris")
    clock = pygame.time.Clock()
    game = Tetris(20, 10)
    if _model_RL == None:
        model_RL = AI.Model_RL(game.width * game.height, 3)
    else:
        model_RL = _model_RL
    counter = 0
    pressing_down = False
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

        if counter % (fps // game.level // 2) == 0 or pressing_down:
            if game.state == "start":
                game.go_down()

        # Make decision by model:
        figure_save = game.get_field_with_figure()
        model_result, action = model_RL.move(figure_save)
        correct_move_flag = 1
        
        if action == 0:
            correct_move_flag = game.go_side(1)
        elif action == 1:
            correct_move_flag = game.go_side(-1)
        elif action == 2:
            correct_move_flag = game.rotate()
        
        # grade made changes
        #print("action: "+ str(action))
        grade, importance = interpreter.evaluate(game, figure_save, np.round(model_result,2), game.score, correct_move_flag)
        model_RL.grade(game.field, grade, importance)
        #print("model_result: "+ str(model_result) + " grade: "+ str(grade))
        #input(" ? ")
        if hidden_mode == False:
            screen.fill(WHITE)
            
            for i in range(game.height):
                for j in range(game.width):
                    pygame.draw.rect(screen, GRAY, [game.x + game.zoom * j, game.y + game.zoom * i, game.zoom, game.zoom], 1)
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

            font = pygame.font.SysFont('Calibri', 25, True, False)
            font1 = pygame.font.SysFont('Calibri', 65, True, False)
            text1 = font.render("Score: " + str(game.score), True, BLACK)
            text2 = font.render("Gameplay: " + str(n_gameplays), True, BLACK)
            text_game_over = font1.render("Game Over", True, (255, 125, 0))
            screen.blit(text1, [0, 0])
            screen.blit(text2, [0, 20])
        if game.state == "gameover":
            if hidden_mode == False:
                screen.blit(text_game_over, [20, 200])
            game.reset()
            n_gameplays -= 1

        if hidden_mode == False: pygame.display.flip()
        if fast_mode == False: clock.tick(fps)
    if hidden_mode == False: pygame.quit()

fps = 4
size = (400, 500)
# start = time.time()
# start_game(size, fps, 3,  _model_RL = None, fast_mode = True, hidden_mode = False)
# end = time.time()
# print("hidden_mode = False", end - start)
# start = time.time()
# start_game(size, fps, 3,  _model_RL = None, fast_mode = True, hidden_mode = True)
# end = time.time()
# print("hidden_mode = True", end - start)

if __name__ == '__main__':
    gameplays = 4
    generation_size = 10
    processes = []
    for i in range(0, generation_size):
        processes.append(Process(target=start_game, args=(size, fps, gameplays)))
    for i in range(0, generation_size):
        processes[i].start()
    for i in range(0, generation_size):
        processes[i].join()