import numpy as np
import helpers

class event:
    def __init__(self, _state, _action, _score_delta, _is_action_corr):
        self.state = _state
        self.action = _action
        self.score_delta = _score_delta
        self.is_action_corr = _is_action_corr
        self.penalty = 0.0
        self.grade = None

class history:
    def __init__(self):
        self.events = []
    
    def create_trainable_set(self):
        X = []
        Y = []
        for i in range(0, min(1000,len(self.events))):
            X.append(helpers.create_model_input(self.events[i].state, False))
            Y.append(self.events[i].grade)
        return np.array(X).T, np.array(Y).T
    

class memory:
    def __init__(self):
        self.memories = []

    def add_history(self, history):
        self.memories.append(history)

    def create_trainable_set(self):
        X = []
        Y = []
        for history in self.memories:
            x, y = history.create_trainable_set()
            if len(X) == 0:
                X = x
                Y = y
            else:
                X = np.concatenate((X, x), axis=1)
                Y = np.concatenate((Y, y), axis=1)

        return X, Y