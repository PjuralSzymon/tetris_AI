import numpy as np
import AI.NeuralNetwork as NN
import copy

class Model_RL:
    def __init__(self, politic_size, actions, empty_mode = False, id = 0):
        hidden_layer_size = (politic_size + actions)
        self.id = id
        self.politic_size = politic_size
        self.actions = actions
        self.last_game_score = -1
        self.M = NN.Model()
        self.punishment_hist = []
        self.importance_hist = []
        if empty_mode: return
        self.M.add_layer(NN.Layer(self.politic_size, hidden_layer_size, NN.Activations.Sigmoid))
        self.M.add_layer(NN.Layer(hidden_layer_size, hidden_layer_size, NN.Activations.Sigmoid))
        self.M.add_layer(NN.Layer(hidden_layer_size, hidden_layer_size, NN.Activations.Sigmoid))
        self.M.add_layer(NN.Layer(hidden_layer_size, hidden_layer_size, NN.Activations.Sigmoid))
        self.M.add_layer(NN.Layer(hidden_layer_size, actions, NN.Activations.SoftMax))

    def summary(self):
        print("politic_size: ", self.politic_size)
        print("actions: ", self.actions)
        self.M.summary()

    def create_input(self, politics):
        input = np.concatenate((politics[0].flatten(), np.array(politics[1:4])))
        input = input.reshape((self.politic_size, 1))
        for i in range(0, len(input)):
            if input[i] > 0:
                input[i] = 1.0
        return input

    def move(self, politics):
        input = self.create_input(politics)
        model_result = self.M.predict_single(input)
        model_result = model_result.transpose()
        return model_result, np.argmax(model_result)

    def grade(self, politics, grade, importance = 1, learning_rate = 0.1):
        input = self.create_input(politics)
        loss_sum = self.M.train(input, grade.transpose(), 1 + int(importance), learning_rate)
        self.punishment_hist.append(loss_sum)

    def save(self, path):
        self.M.save(path)

    def load(path):
        model = Model_RL(1,1, True)
        model.M.load(path)
        model.politic_size = model.M.Layers[0].W.shape[1]
        model.actions = model.M.Layers[-1].W.shape[0]
        return model

    def deepcopy(self, diffrence_rate):
        model = Model_RL(self.politic_size, self.actions, True)
        model.M = self.M.deepcopy(diffrence_rate)
        model.last_game_score = 0
        return model
