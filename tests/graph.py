import matplotlib.pyplot as plt

def sum(array):
    result = 0
    for i in range(0, len(array)):
        array[i] = array[i].replace('[', '')
        array[i] = array[i].replace(']', '')
        array[i] = array[i].replace(',', '')
        result += float(array[i])
    return result
    
# Wczytaj dane z pliku
with open("./history.txt", 'r') as f:
    lines = f.readlines()

# Przetwórz dane w formacie:
# epoch:  0  best_models:  [4.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0]  score in epoch:  [1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0, 1.0, 0.0]
best_models = []
scores = []
for line in lines:
    parts = line.split()
    best_models.append(sum(parts[3:11]))
    scores.append(sum(parts[14:29]))

# Stwórz pierwszy wykres
plt.plot(best_models, label='best_models')
plt.xlabel('epoch')
plt.ylabel('best_models')
plt.title('Wykres best_models')
plt.legend()
plt.show()

# Stwórz drugi wykres
plt.plot(scores, label='scores')
plt.xlabel('epoch')
plt.ylabel('scores')
plt.title('Wykres scores')
plt.legend()
plt.show()