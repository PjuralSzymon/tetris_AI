#nie zmienialne:
fps = 2
size = (400, 500)
#zmienialne:
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
GAME_WIDTH = 10
GAME_HIGHT = 20
NOISE = 0.225
NOISE_END_VAL = 0.0058
IMPORTANCE_THRESH = 0.154
importance_height_sum_max = 0.7479
importance_height_sum_stdev = 0.2811
importance_height_dist_max = 0.3275
importance_height_dist_mean = 0.2876
learning_rate = 0.19881703
punishment = 0.5263

#P4_195 wybralem te konfiguracje


class cf:
    # ta klasa jest potrzebna poniewaz algorytmy korzystaja z wieowatkowosci wiec zeby wysylac im dane
    # nie tracąc mozliwosci edycji pliku config tworzona jest kopia przesylana do procesowj ako obiekt
    def __init__(self):
        self.fps = fps
        self.size = size
        self.RED = RED
        self.BLACK = BLACK
        self.WHITE = WHITE
        self.GRAY = GRAY
        self.GAME_WIDTH = GAME_WIDTH
        self.GAME_HIGHT = GAME_HIGHT
        self.NOISE = NOISE
        self.NOISE_END_VAL = NOISE_END_VAL
        self.IMPORTANCE_THRESH = IMPORTANCE_THRESH
        self.punishment = punishment
        self.importance_height_sum_max = importance_height_sum_max
        self.importance_height_sum_stdev = importance_height_sum_stdev
        self.importance_height_dist_max = importance_height_dist_max
        self.importance_height_dist_mean = importance_height_dist_mean
        self.learning_rate = learning_rate

def get_config_object():
    return cf()
