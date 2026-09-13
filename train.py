from utils import randInitializeWeights, prediction, gradientDescent
import numpy as np 
import pandas as pd


data = pd.read_csv("processed_data.csv", header=None)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
X = data.drop(columns=data.columns[-1])
y = data.drop(columns=data.columns[:-1])

pct_valid = 0.04
pct_teste = 0.2

N_TOTAL = X.shape[0]
NUM_TREINO = int((1 - pct_teste) * N_TOTAL)
NUM_VALID = int(pct_valid * N_TOTAL)


X_treino, y_treino = X.iloc[:NUM_TREINO], y.iloc[:NUM_TREINO]

X_valid, y_valid = X.iloc[N_TOTAL - NUM_VALID:], y.iloc[N_TOTAL-NUM_VALID:]

X_teste, y_teste = X.iloc[NUM_TREINO : N_TOTAL - NUM_VALID], y.iloc[NUM_TREINO : N_TOTAL - NUM_VALID]

print(len(X_treino), len(y_treino))
print(len(X_teste), len(y_teste))
print(len(X_valid), len(y_valid))


input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

initial_theta1 = randInitializeWeights(input_layer_size,hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size,num_labels)
initial_theta = np.append(initial_theta1.flatten(),initial_theta2.flatten())

theta,J_history = gradientDescent(X_treino,y_treino,initial_theta,0.8,800,1,input_layer_size,hidden_layer_size,num_labels)
theta1 = theta[:((input_layer_size+1)*hidden_layer_size)].reshape(hidden_layer_size,input_layer_size+1)
theta2 = theta[((input_layer_size+1)*hidden_layer_size):].reshape(num_labels,hidden_layer_size+1)

X_teste = X_teste.to_numpy()
y_teste = y_teste.to_numpy()

pred = prediction(X_teste,theta1,theta2)

pred = np.asarray(pred).ravel()
y_teste_flat = y_teste.ravel()

print("Training Set Accuracy:", np.mean(pred == y_teste_flat) * 100, "%")