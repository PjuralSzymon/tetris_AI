import numpy as np

def move(politics):
    input = np.array(politics)
    print(input.shape)
    # here model do something ...
    result = np.random.rand(4)
    # 0 - do nothing
    # 1 - rotate
    # 2 - move right
    # 3 - move left
    sum = np.sum(result)
    result = result/sum
    return result, np.argmax(result)