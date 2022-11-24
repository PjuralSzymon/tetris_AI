import numpy as np
import AI.NeuralNetwork as NN


class Model_RL:
    def __init__(self, politic_size, actions):
        hidden_layer_size = int((politic_size + actions)/10)
        self.politic_size = politic_size
        self.actions = actions
        self.M = NN.Model()
        self.M.add_layer(NN.Layer(politic_size, hidden_layer_size, NN.Activations.ReLu))
        self.M.add_layer(NN.Layer(hidden_layer_size, hidden_layer_size, NN.Activations.ReLu))
        self.M.add_layer(NN.Layer(hidden_layer_size, actions, NN.Activations.SoftMax))

    def create_input(self, politics):
        input = np.array(politics).flatten()
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

    def grade(self, politics, grade):
        input = self.create_input(politics)
        self.M.train(input, grade.transpose(), 0.1)
