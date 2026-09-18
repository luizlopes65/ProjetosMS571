from math import ceil, sqrt
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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


def plot_activation_images(theta1):
    num_neurons = theta1.shape[0]
    fig, axes = create_activation_grid(num_neurons)

    for neuron in range(num_neurons):
        ax = axes[neuron]
        image = theta1[neuron, 1:].reshape(20, 20)
        ax.imshow(normalize(image), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Neurônio {neuron + 1}")
        ax.axis("off")

    for ax in axes[num_neurons:]:
        ax.axis("off")

    fig.suptitle("Pesos finais dos neurônios escondidos")
    plt.tight_layout()
    plt.show()


def create_activation_gif(
    snapshots, output_path=VISUALIZATIONS_DIR / "activation_evolution.gif"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_neurons = snapshots[0][1].shape[0]
    fig, axes = create_activation_grid(num_neurons)
    images = []

    for neuron in range(num_neurons):
        ax = axes[neuron]
        image = snapshots[0][1][neuron, 1:].reshape(20, 20)
        images.append(ax.imshow(normalize(image), cmap="gray", vmin=0, vmax=1))
        ax.set_title(f"Neurônio {neuron + 1}")
        ax.axis("off")

    for ax in axes[num_neurons:]:
        ax.axis("off")

    def update(frame):
        iteration, theta1_snapshot, _ = snapshots[frame]

        for neuron, image_plot in enumerate(images):
            image = theta1_snapshot[neuron, 1:].reshape(20, 20)
            image_plot.set_data(normalize(image))

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


if __name__ == "__main__":
    X_treino, y_treino, _, _, _, _ = get_data_partitioned()
    theta1, _, _, _, snapshots = train_model(
        X_treino,
        y_treino,
        snapshot_every=10,
    )
    plot_activation_images(theta1)
    create_activation_gif(snapshots)
    print("GIF salvo em activation_evolution.gif")
