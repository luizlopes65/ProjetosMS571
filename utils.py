import numpy as np
from tqdm import tqdm


def validate_layer_sizes(layer_sizes):
    """Valida a arquitetura e retorna seus tamanhos como uma tupla."""
    if layer_sizes is None:
        raise ValueError("layer_sizes não pode ser None.")

    sizes = tuple(layer_sizes)
    if len(sizes) < 2:
        raise ValueError("layer_sizes deve conter entrada e saída.")
    if any(
        isinstance(size, bool)
        or not isinstance(size, (int, np.integer))
        or size <= 0
        for size in sizes
    ):
        raise ValueError("Todos os tamanhos de camada devem ser inteiros positivos.")
    return sizes


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoidGradient(z):
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid*(1-sigmoid)


def unpack_thetas(theta, layer_sizes):
    """Desempacota o vetor 1D ``theta`` em uma lista de matrizes de peso,
    uma por transição entre camadas consecutivas de ``layer_sizes``
    (layer_sizes[0] = entrada, layer_sizes[-1] = número de classes)."""
    layer_sizes = validate_layer_sizes(layer_sizes)
    theta = np.asarray(theta, dtype=float).ravel()
    expected_size = sum(
        l_out * (l_in + 1)
        for l_in, l_out in zip(layer_sizes[:-1], layer_sizes[1:])
    )
    if theta.size != expected_size:
        raise ValueError(
            f"theta possui {theta.size} valores, mas a arquitetura exige "
            f"{expected_size}."
        )

    thetas = []
    offset = 0
    for l_in, l_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        size = l_out * (l_in + 1)
        thetas.append(theta[offset:offset + size].reshape(l_out, l_in + 1))
        offset += size
    return thetas


def pack_thetas(thetas):
    """Empacota a lista de matrizes de peso de volta em um único vetor 1D."""
    return np.concatenate([t.ravel() for t in thetas])


def randInitializeWeights(L_in, L_out):
    epi = (6**(1/2))/(L_in+L_out)**(1/2)
    W = np.random.rand(L_out, L_in+1)*(2*epi)-epi
    return W


def randInitializeAllWeights(layer_sizes):
    """Inicializa aleatoriamente a lista completa de pesos, uma matriz por
    transição entre camadas consecutivas de ``layer_sizes``."""
    layer_sizes = validate_layer_sizes(layer_sizes)
    return [randInitializeWeights(l_in, l_out)
            for l_in, l_out in zip(layer_sizes[:-1], layer_sizes[1:])]


def computeCost(X, y, theta, layer_sizes, num_labels, Lambda):
    layer_sizes = validate_layer_sizes(layer_sizes)
    if num_labels != layer_sizes[-1]:
        raise ValueError("num_labels deve ser igual ao tamanho da camada de saída.")
    thetas = unpack_thetas(theta, layer_sizes)
    L = len(thetas)

    m = X.shape[0]
    J = 0
    X = np.hstack((np.ones((m,1)),X))
    y10 = np.zeros((m, num_labels))

    # Passada para frente (generalizada para N camadas)
    a = [X]                                    # a[0] = entrada com bias
    for l, theta_l in enumerate(thetas):
        a_next = sigmoid(a[-1] @ theta_l.T)
        if l < L-1:                            # camadas escondidas ganham bias
            a_next = np.hstack((np.ones((m,1)),a_next))
        a.append(a_next)
    a2 = a[-1]                                 # saída (sem bias)

    for i in range(1, num_labels+1):
        y10[:,i-1][:,np.newaxis] = np.where(y==i,1,0)
    for j in range(num_labels):
        J = J + sum(-y10[:,j]*np.log(a2[:,j])-(1-y10[:,j])*np.log(1-a2[:,j]))

    cost = 1/m*J
    reg_J = cost + Lambda/(2*m)*sum(np.sum(theta_l[:,1:]**2) for theta_l in thetas)

    # Backpropagation (por amostra, generalizado para N camadas)
    grad = [np.zeros((theta_l.shape)) for theta_l in thetas]
    for i in range(m):
        xi = X[i,:]
        a_i = [a_k[i,:] for a_k in a]          # ativações desta amostra
        d = [None]*L
        d[L-1] = a_i[-1] - y10[i,:]            # d2 = a2i - y10[i,:]
        for l in range(L-2, -1, -1):
            d[l] = thetas[l+1].T @ d[l+1] * sigmoidGradient(np.hstack((1, a_i[l] @ thetas[l].T)))
            d[l] = d[l][1:]                    # descarta o bias, como d1[1:] no original
        for l in range(L):
            grad[l] = grad[l] + d[l][:,np.newaxis] @ a_i[l][:,np.newaxis].T

    grad = [g/m for g in grad]
    grad_reg = [g + (Lambda/m)*np.hstack((np.zeros((theta_l.shape[0],1)), theta_l[:,1:]))
                for g, theta_l in zip(grad, thetas)]

    return cost, grad, reg_J, grad_reg


def gradientDescent(X, y, theta, alpha, nbr_iter, Lambda, layer_sizes, snapshot_callback=None):
    layer_sizes = validate_layer_sizes(layer_sizes)
    thetas = unpack_thetas(theta, layer_sizes)
    J_history = []

    for i in tqdm(range(nbr_iter)):
        theta = pack_thetas(thetas)
        cost, grad = computeCost(X, y, theta, layer_sizes, layer_sizes[-1], Lambda)[2:]  # regularizados
        thetas = [t - (alpha*g) for t, g in zip(thetas, grad)]
        J_history.append(cost)

        if snapshot_callback is not None:
            snapshot_callback(i + 1, [t.copy() for t in thetas])

    nn_paramsFinal = pack_thetas(thetas)
    return nn_paramsFinal, J_history


def prediction(X, theta):
    m = X.shape[0]
    X = np.hstack((np.ones((m,1)),X))

    a = X
    for layer_index, theta_l in enumerate(theta):
        a = sigmoid(a @ theta_l.T)
        if layer_index < len(theta) - 1:
            a = np.hstack((np.ones((m,1)),a))

    return np.argmax(a,axis=1)+1
