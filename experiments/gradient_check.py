"""Checagem do gradiente (gradient checking) do backpropagation.

Verifica se os gradientes analíticos de ``utils.computeCost`` estão corretos
comparando-os com uma aproximação numérica por diferenças finitas, em uma rede
pequena com exemplos fictícios (valores aleatórios) — como pedido no PDF.

O teste é rodado com 1 camada escondida (a arquitetura sugerida no PDF,
3-5-3) e também com 2 camadas escondidas (3-5-4-3), que é o único caso que
exercita a propagação do erro entre camadas intermediárias.

Uso:
    python experiments/gradient_check.py            # usa as arquiteturas padrão
    python experiments/gradient_check.py 3 5 4 3    # ou passe sua arquitetura
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import computeCost, pack_thetas, randInitializeAllWeights

EPSILON = 1e-4


def compute_numerical_gradient(cost_fn, theta, epsilon=EPSILON):
    """Aproxima o gradiente de ``cost_fn`` por diferenças finitas:
    (J(theta + eps*e_i) - J(theta - eps*e_i)) / (2*eps), para cada posição i."""
    numgrad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_plus[i] += epsilon
        theta_minus = theta.copy()
        theta_minus[i] -= epsilon
        numgrad[i] = (cost_fn(theta_plus) - cost_fn(theta_minus)) / (2 * epsilon)
    return numgrad


def relative_difference(a, b):
    """Diferença relativa entre dois vetores (critério do material do curso)."""
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b))


def gradient_check(
    layer_sizes,
    m=3,
    Lambda=1,
    epsilon=EPSILON,
    seed=0,
    tol=1e-7,
):
    """Compara o gradiente analítico (backprop) com o numérico numa rede pequena.

    ``layer_sizes`` ex.: [3, 5, 3] (entrada, escondida(s), saída).
    """
    input_layer_size, num_labels = layer_sizes[0], layer_sizes[-1]

    # Exemplos de treinamento fictícios (valores aleatórios)
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(m, input_layer_size))
    y = rng.integers(1, num_labels + 1, size=(m, 1))

    # Inicialização dos pesos
    np.random.seed(seed)
    thetas = randInitializeAllWeights(layer_sizes)
    theta = pack_thetas(thetas)

    # Função de custo apenas em função de theta (com regularização)
    cost_fn = lambda th: computeCost(X, y, th, layer_sizes, num_labels, Lambda)[2]
    analytic = pack_thetas(computeCost(X, y, theta, layer_sizes, num_labels, Lambda)[3])

    numgrad = compute_numerical_gradient(cost_fn, theta, epsilon)

    diff = relative_difference(analytic, numgrad)
    ok = diff < tol

    print(f"Arquitetura {layer_sizes} | m={m} | lambda={Lambda} | eps={epsilon}")
    print(f"  Gradiente analítico (backprop)  : norma = {np.linalg.norm(analytic):.6e}")
    print(f"  Gradiente numérico              : norma = {np.linalg.norm(numgrad):.6e}")
    print(f"  Diferença relativa = {diff:.3e}  ->  {'OK' if ok else 'FALHOU'}")
    return ok


def run_default_checks():
    checks = [
        gradient_check([3, 3]),       # sem camada escondida
        gradient_check([3, 5, 3]),    # 1 camada escondida
        gradient_check([3, 5, 4, 3]), # 2 camadas escondidas
    ]
    passed = all(checks)
    print("\nRESULTADO:", "TODOS OS TESTES PASSARAM" if passed else "HA FALHAS")
    return passed


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    if args:
        ok = gradient_check(args)
        print("\nRESULTADO:", "OK" if ok else "FALHOU")
        sys.exit(0 if ok else 1)
    sys.exit(0 if run_default_checks() else 1)
