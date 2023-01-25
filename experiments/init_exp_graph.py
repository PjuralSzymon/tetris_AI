import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_file(file):
    def F(x):
        x = x.replace(' ', '')
        return float(x)
    def I(x):
        x = x.replace(' ', '')
        return float(x)
    data = []
    with open(file) as f:
        for line in f:
            line = line.replace('imp_h_sum_ma', ' ')
            line_data = line.strip().split(", ")
            line_data_2 = line_data[2].split(" ")
            data.append([line_data[0], I(line_data[1]), F(line_data_2[1]), F(line_data_2[3]), F(line_data_2[5]), F(line_data_2[7])])
    df = pd.DataFrame(data, columns=["ID", "result", "noise", "noise_ev", "imp_thr", "punish"])
    return df

# lista z nazwami plików
files = ["../results/init_experiment_2/result_history.txt"]
# "../results/init_experiment/P3_result_history.txt", 
# "../results/init_experiment/result_history_18012023.txt",
# "../results/init_experiment/result_history_19012023.txt"]

# lista z najlepszymi wynikami
df_merge = []

for file in files:
    df = read_file(file)
    print("df.length: ", df.shape)
    df_merge.append(df)
df = pd.concat(df_merge)
print(df.info())
print(df.head())
print(df.tail())

def draw_graph(column, x_label, y_label):
    x = range(0, len(column))
    mean = np.mean(np.array(column))
    print("mean ", y_label, ": ", mean)
    print("first ", y_label, ": ", column.iloc[0])
    plt.axhline(y=mean, color='r', linestyle='-')
    # wykres kolumnowy dla result
    plt.bar(x, column)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

draw_graph(df['result'], "agent id number", "score")
# wykres 30 najlepszych wyników result
best_results = df.nlargest(5, "result")
draw_graph(best_results['result'], "agent id number", " best scores")
draw_graph(best_results['noise'], "agent id number", "noise")
draw_graph(best_results['noise_ev'], "agent id number", "noise lower bound")
draw_graph(best_results['imp_thr'], "agent id number", "importance thresh")
draw_graph(best_results['punish'], "agent id number", "punishment")