from math import ceil, sqrt
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import get_data_partitioned, train_model

VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"


def normalize(image):
    image_min, image_max = image.min(), image.max()
    if image_max == image_min:
        return image
    return (image - image_min) / (image_max - image_min)


def create_activation_grid(num_neurons):
    num_columns = ceil(sqrt(num_neurons))
    num_rows = ceil(num_neurons / num_columns)
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(3 * num_columns, 3 * num_rows),
        squeeze=False,
    )
    return fig, list(axes.flat)


def _get_layer(thetas, layer_index):
    if not thetas:
        raise ValueError("thetas deve conter pelo menos uma matriz de pesos.")
    if not 0 <= layer_index < len(thetas):
        raise ValueError(
            f"layer_index deve estar entre 0 e {len(thetas) - 1}; "
            f"recebido: {layer_index}."
        )
    return np.asarray(thetas[layer_index], dtype=float)


def _plot_weight(weights, input_shape=None):
    weights = np.asarray(weights, dtype=float)
    if input_shape is not None and np.prod(input_shape) == weights.size:
        return normalize(weights.reshape(input_shape)), "image"
    return normalize(weights[np.newaxis, :]), "weights"


def plot_activation_images(
    thetas,
    layer_index=0,
    input_shape=None,
    output_path=None,
    show=True,
):
    """Visualiza os pesos de qualquer transição da rede.

    A primeira camada pode ser mostrada como imagem quando ``input_shape``
    corresponde ao número de entradas. Para outras arquiteturas, cada vetor
    de pesos é mostrado como um mapa de calor genérico.
    """
    theta = _get_layer(thetas, layer_index)
    num_neurons = theta.shape[0]
    fig, axes = create_activation_grid(num_neurons)

    for neuron, axis in enumerate(axes[:num_neurons]):
        image, image_type = _plot_weight(theta[neuron, 1:], input_shape)
        axis.imshow(image, cmap="gray" if image_type == "image" else "coolwarm", aspect="auto")
        axis.set_title(f"Neurônio {neuron + 1}")
        axis.axis("off")

    for axis in axes[num_neurons:]:
        axis.axis("off")

    fig.suptitle(f"Pesos da transição {layer_index + 1}")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Figura salva em: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def create_activation_gif(
    snapshots,
    output_path=VISUALIZATIONS_DIR / "activation_evolution.gif",
    layer_index=0,
    input_shape=None,
):
    """Gera um GIF com a evolução dos pesos de uma transição escolhida."""
    if not snapshots:
        raise ValueError("snapshots deve conter pelo menos um estado da rede.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_thetas = snapshots[0][1]
    first_theta = _get_layer(first_thetas, layer_index)
    num_neurons = first_theta.shape[0]
    fig, axes = create_activation_grid(num_neurons)
    images = []

    for neuron, axis in enumerate(axes[:num_neurons]):
        image, image_type = _plot_weight(first_theta[neuron, 1:], input_shape)
        images.append(
            axis.imshow(
                image,
                cmap="gray" if image_type == "image" else "coolwarm",
                aspect="auto",
            )
        )
        axis.set_title(f"Neurônio {neuron + 1}")
        axis.axis("off")

    for axis in axes[num_neurons:]:
        axis.axis("off")

    def update(frame):
        iteration, thetas_snapshot = snapshots[frame]
        theta = _get_layer(thetas_snapshot, layer_index)
        for neuron, image_plot in enumerate(images):
            image, _ = _plot_weight(theta[neuron, 1:], input_shape)
            image_plot.set_data(image)
        fig.suptitle(f"Evolução dos pesos — iteração {iteration}")
        return images

    animation = FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=100,
        blit=False,
    )
    animation.save(output_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"GIF salvo em: {output_path}")


if __name__ == "__main__":
    X_treino, y_treino, _, _, _, _ = get_data_partitioned()
    thetas, _, _, snapshots = train_model(
        X_treino,
        y_treino,
        snapshot_every=10,
    )
    plot_activation_images(thetas, input_shape=(20, 20))
    create_activation_gif(snapshots, input_shape=(20, 20))
