import numpy as np
import AI.NeuralNetwork as NN
import AI.Memory
import helpers


class Model_RL:
    def __init__(self, politic_size, actions, empty_mode = False):
        hidden_layer_size = int((politic_size + actions)/2)
        self.memory = AI.Memory.memory()
        self.politic_size = politic_size
        self.actions = actions
        self.M = NN.Model()
        if empty_mode: return
        self.M.add_layer(NN.Layer(politic_size, politic_size, NN.Activations.Sigmoid))
        self.M.add_layer(NN.Layer(politic_size, hidden_layer_size, NN.Activations.Sigmoid))
        #self.M.add_layer(NN.Layer(hidden_layer_size, hidden_layer_size, NN.Activations.Sigmoid))
        #self.M.add_layer(NN.Layer(hidden_layer_size, hidden_layer_size, NN.Activations.Sigmoid))
        #self.M.add_layer(NN.Layer(hidden_layer_size, hidden_layer_size, NN.Activations.Sigmoid))
        self.M.add_layer(NN.Layer(hidden_layer_size, actions, NN.Activations.SoftMax))

    def summary(self):
        print("politic_size: ", self.politic_size)
        print("actions: ", self.actions)
        self.M.summary()



    def learn(self, epochs = 3):
        x_train, y_train = self.memory.create_trainable_set()
        acc2 = self.M.train(x_train, y_train, epochs, 0.1)
        #print("acc2: ", acc2)
        return acc2#self.M.train(x_train, y_train, 10, 0.1)

    def move(self, politics):
        #input = self.create_input(politics)
        input = helpers.create_model_input(politics, True)
        model_result = self.M.predict_single(input)
        model_result = model_result.transpose()
        return model_result, np.argmax(model_result)

    def grade(self, politics, grade, importance = 1):
        #input = self.create_input(politics)
        input = helpers.create_model_input(politics, True)
        self.M.train(input, grade.transpose(), importance, 0.1)

    def save(self, path):
        self.M.save(path)

    def load(path):
        model = Model_RL(1,1, True)
        model.M.load(path)
        model.politic_size = model.M.Layers[0].W.shape[0]
        model.actions = model.M.Layers[-1].W.shape[0]
        return model

    def deepcopy(self, diffrence_rate):
        model = Model_RL(self.politic_size, self.actions, True)
        model.M = self.M.deepcopy(diffrence_rate)
        return model
