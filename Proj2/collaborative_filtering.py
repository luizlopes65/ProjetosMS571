import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

from utils import cofi_cost_function, normalize_ratings

def main():
    movie_list = []
    with open("data/movie_names.txt", 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            movie_list.append(parts[1])

    data = loadmat('data/movie_data.mat')
    Y = data['Y']
    R = data['R']

    n_m, n_u = Y.shape
    n_features = 100
    _lambda = 0.0

    np.random.seed(42)
    X = np.random.randn(n_m, n_features) * 0.01
    Theta = np.random.randn(n_u, n_features) * 0.01

    Y_norm, Y_mean = normalize_ratings(Y, R)
    initial_parameters = np.concatenate((X.ravel(), Theta.ravel()))
    
    result = minimize(
        fun=cofi_cost_function,
        x0=initial_parameters,
        args=(Y_norm, R, n_u, n_m, n_features, _lambda),
        method='CG',
        jac=True,
        options={'maxiter': 100, 'disp': True}
    ) 
    # O prof pediu pra usar o gradiente conjugado então eu usei a função do scipy, 
    # mas não sei se ele quer que preserve aquela ideia de atualizar Theta e X
    # alternadamente, pq acho que essa função faz tudo ao mesmo tempo

    optimized_parameters = result.x
    X_opt = optimized_parameters[:(n_m * n_features)].reshape(n_m, n_features)
    Theta_opt = optimized_parameters[(n_m * n_features):].reshape(n_u, n_features)

    predict = X_opt @ Theta_opt.T + Y_mean

    movie_mean_score = np.mean(predict, axis=1)

    top_10 = np.argsort(movie_mean_score)[-10:][::-1]

    for i, idx in enumerate(top_10, start=1):
        nome_filme = movie_list[idx] 
        print(f"{i:2d}º - {nome_filme} (Nota média: {movie_mean_score[idx]:.2f})")

if __name__ == "__main__":
    main()