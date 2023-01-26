import random
import copy
import numpy as np

"kod do tetrisa z https://levelup.gitconnected.com/writing-tetris-in-python-2a16bddb5318"


colors = [
    (0, 0, 0),
    (120, 37, 179),
    (100, 179, 179),
    (80, 34, 22),
    (80, 134, 22),
    (180, 34, 22),
    (180, 34, 122),
]


class Figure:
    x = 0
    y = 0

    figures = [
        [[1, 5, 9, 13], [4, 5, 6, 7]],
        # [[4, 5, 9, 10], [2, 6, 5, 9]],
        # [[6, 7, 9, 10], [1, 5, 6, 10]],
        # [[1, 2, 5, 9], [0, 4, 5, 6], [1, 5, 9, 8], [4, 5, 6, 10]],
        # [[1, 2, 6, 10], [5, 6, 7, 9], [2, 6, 10, 11], [3, 5, 6, 7]],
        # [[1, 4, 5, 6], [1, 4, 5, 9], [4, 5, 6, 9], [1, 5, 6, 9]],
        [[1, 2, 5, 6]],
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = random.randint(0, len(self.figures) - 1)
        self.color = random.randint(1, len(colors) - 1)
        self.rotation = 0

    def image(self):
        return self.figures[self.type][self.rotation]

    def rotate(self):
        self.rotation = (self.rotation + 1) % len(self.figures[self.type])


class Tetris:
    level = 2
    score = 0
    state = "start"
    field = []
    height = 0
    width = 0
    x = 100
    y = 60
    zoom = 20
    figure = None

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.field = []
        self.score = 0
        self.state = "start"
        for i in range(height):
            new_line = []
            for j in range(width):
                new_line.append(0)
            self.field.append(new_line)

    def new_figure(self):
        self.figure = Figure(3, 0)

    def intersects(self):
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection

    def break_lines(self):
        lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += lines ** 2

    def go_space(self):
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze()

    def go_down(self):
        self.figure.y += 1
        if self.intersects():
            self.figure.y -= 1
            self.freeze()

    def freeze(self):
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = self.figure.color
        self.break_lines()
        self.new_figure()
        if self.intersects():
            self.state = "gameover"

    def get_field_no_figure(self):
        return np.array(copy.deepcopy(self.field))

    def get_field_with_figure(self):
        field_copy = copy.deepcopy(self.field)
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    field_copy[i + self.figure.y][j + self.figure.x] = self.figure.color
        return np.array(field_copy)

    def go_side(self, dx):
        correct_move = 1
        old_x = self.figure.x
        self.figure.x += dx
        if self.intersects():
            self.figure.x = old_x
            correct_move = 0
        return correct_move

    def rotate(self):
        correct_move = 1
        old_rotation = self.figure.rotation
        self.figure.rotate()
        if self.intersects():
            self.figure.rotation = old_rotation
            correct_move = 0
        return correct_move

    def reset(self):
        self.__init__(self.height, self.width)
# for player laying:
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         done = True
    #     if event.type == pygame.KEYDOWN:
    #         if event.key == pygame.K_UP:
    #             game.rotate()
    #         if event.key == pygame.K_DOWN:
    #             pressing_down = True
    #         if event.key == pygame.K_LEFT:
    #             game.go_side(-1)
    #         if event.key == pygame.K_RIGHT:
    #             game.go_side(1)
    #         if event.key == pygame.K_SPACE:
    #             game.go_space()
    #         if event.key == pygame.K_ESCAPE:
    #             game.__init__(20, 10)

    # if event.type == pygame.KEYUP:
    #         if event.key == pygame.K_DOWN:
    #             pressing_down = False