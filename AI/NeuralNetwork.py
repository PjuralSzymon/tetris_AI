import numpy as np
import random
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

    # def train(self, X, E, alpha):
    #     self.predict_single(X)
    #     m = E.size
    #     for layer in reversed(self.Layers):
    #         E, _ = layer.back_prop(E,m)
    #     for layer in self.Layers:
    #         layer.update_params(alpha)

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