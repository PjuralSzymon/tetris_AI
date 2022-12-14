import numpy as np
import random
import helpers
#from helpers import *

class Loss:
    class MSE:
        def exe(Y_true, Y_pred):
            return np.mean(np.power(Y_true-Y_pred, 2))
        def der(Y_true, Y_pred):
            return np.array(2*(Y_pred-Y_true)/Y_true.size)
    class MAE:
        def exe(Y_true, Y_pred):
            return np.mean(np.abs(Y_true-Y_pred))
        def der(Y_true, Y_pred):
            N = Y_true.shape[0]
            return np.array(-((Y_true - Y_pred) / (abs(Y_true - Y_pred)))/N)

class Activations:
    class ReLu:
        def exe(X):
            return np.maximum(X,0)
        def der(X):
            return X > 0
    class Sigmoid:
        def exe(X):
            return 1/(1+np.exp(-X))
        def der(X):
            s = Activations.Sigmoid.exe(X)
            return s * (1 - s)
    class SoftMax:
        def exe(X):
            X[X== np.inf] = np.nan
            X[X==-np.inf] = np.nan
            exp = np.exp(X - np.nanmax(X))
            return np.nan_to_num(exp / exp.sum(axis=0), nan = np.nanmean(X))
        def der(X):
            return 1
            


class Layer:
    def __init__(self,input_size, neurons, act_fun):
        self.input_size = input_size
        self.neurons = neurons
        self.act_fun = act_fun
        self.W = np.random.rand(neurons, input_size) - 0.5
        self.B = np.random.rand(neurons, 1) - 0.5

    def deepcopy(self, diffrence_rate):
        new_layer = Layer(self.input_size, self.neurons, self.act_fun)
        new_layer.W = helpers.randomize_matrix(self.W, diffrence_rate)
        new_layer.B = helpers.randomize_matrix(self.B, diffrence_rate)
        return new_layer

    def forward_prop(self, X):
        self.I = X
        self.Z = self.W.dot(X) + self.B
        self.A = self.act_fun.exe(self.Z)
        return self.A

    def back_prop(self,E):
        dZ = self.act_fun.der(self.Z) * E
        self.dW = np.dot(dZ, self.I.T)
        self.dB = np.mean(dZ, axis=1)
        E = np.dot(self.W.T, dZ)
        return E

    def update_params(self, alpha):
        self.W = self.W - alpha * self.dW
        self.B = self.B - alpha * np.array([self.dB]).T
        #self.W = Layer.clean_nan(self.W)
        #self.B = Layer.clean_nan(self.B)

    def clean_nan(x):
        x[x== np.inf] = np.nan
        x[x==-np.inf] = np.nan
        x = np.nan_to_num(x, random.uniform(-0.1, 0.1))
        return x

    def to_str(self):
        result = ""
        result += str(self.act_fun.__name__) + '\n'
        result += str(self.W.shape[0]) + '\n'
        result += str(self.W.shape[1]) + '\n'
        result += helpers.numpy2str(self.W) + '\n'
        result += str(self.B.shape[0]) + '\n'
        result += str(self.B.shape[1]) + '\n'
        result += helpers.numpy2str(self.B) + '\n'
        return result

class Model:
    def __init__(self):
        self.Layers = []
        pass
    
    def add_layer(self, Layer):
        self.Layers.append(Layer)

    def predict_single(self, X):
        for layer in self.Layers:
            X = layer.forward_prop(X)
        return X

    def get_predictions(A2):
        return np.argmax(A2, 0)

    def get_accuracy(predictions, Y):
        return np.sum(predictions == Y) / Y.size


    def train(self, X, Y, epochs, alpha):
        for i in range(0, epochs):
            A = self.predict_single(X)
            E = Loss.MAE.der(Y, A)
            loss = Loss.MAE.exe(Y, A)
            for layer in reversed(self.Layers):
                E = layer.back_prop(E)
            for layer in self.Layers:
                layer.update_params(alpha)
        #print("loss, ", loss)

    def summary(self):
        id = 0 
        for layer in self.Layers:
            print("id: ", id)
            print("layer.W: ", layer.W.shape, " head: ", round(layer.W[0][0],2), round(layer.W[2][2],2), round(layer.W[3][3],2), " tail: ", layer.W[-1][-1])
            print("layer.B: ", layer.B.shape, " head: ", round(layer.B[0][0],2), round(layer.B[2][0],2), round(layer.B[3][0],2), " tail: ", layer.B[-1][-1])
            print("layer A:", layer.act_fun)
            print("---")
            id += 1

    def save(self, path):
        text_file = open(path, "w")
        text = ""
        for layer in self.Layers:
            text += layer.to_str()
        text_file.write(text)
        text_file.close()

    def deepcopy(self, diffrence_rate):
        M = Model()
        for L in self.Layers:
            M.add_layer(L.deepcopy(diffrence_rate))
        return M

    def load(self, path):
        text_file = open(path, "r")
        count = 0
        while True:
            count += 1
            act_fun_label = text_file.readline()
            if not act_fun_label:
                break
            if 'Sigmoid' in act_fun_label:
                act_fun = Activations.Sigmoid
            elif 'ReLu' in act_fun_label:
                act_fun = Activations.ReLu
            elif 'SoftMax' in act_fun_label:
                act_fun = Activations.SoftMax
            input_size = int(text_file.readline())
            neurons = int(text_file.readline())
            W = helpers.str2numpy(text_file.readline(), input_size, neurons)
            bias_x = int(text_file.readline())
            bias_y = int(text_file.readline())
            B = helpers.str2numpy(text_file.readline(), bias_x, bias_y)
            W = np.array(W)
            B = np.array(B)
            L = Layer(input_size, neurons, act_fun)
            L.W = W
            L.B = B
            self.add_layer(L)
        text_file.close()

# data = pd.read_csv('C:\\Uczymy sie\\_Magisterka\\NeuralNetworks\\dataset\\train.csv')

# data = np.array(data)
# m,n = data.shape
# np.random.shuffle(data)

# data_test = data[0:1000].T
# Y_test = data_test[0]
# X_test = data_test[1:n]
# X_test = X_test / 255.0

# data_train = data[1000:m].T
# Y_train = data_train[0]
# X_train = data_train[1:n]
# X_train = X_train / 255.0

# M = Model()
# M.add_layer(Layer(784,10,Activations.ReLu))
# M.add_layer(Layer(10,10,Activations.ReLu))
# M.add_layer(Layer(10,10,Activations.SoftMax))
# M.gradient_descent(X_train, Y_train, 10, 0.1)

# predicted = M.predict_single(X_train[:, 2,None])
# print(predicted)