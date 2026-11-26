"""Gera gráficos, métricas e rankings da filtragem colaborativa."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from collaborative_filtering import TOP_N, load_movie_data, predict_ratings

OUTPUT_DIR = Path(__file__).parent / "results" / "recommender"
MIN_RATINGS_FILTER = 20

plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
})


def br(valor, casas=2):
    """Formata número com vírgula decimal, como no relatório."""
    return f"{valor:.{casas}f}".replace(".", ",")


def cientifica(valor, casas=2):
    """Formata em notação científica com vírgula decimal e expoente sobrescrito."""
    mantissa, expoente = f"{valor:.{casas}e}".split("e")
    sobrescrito = str(int(expoente)).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    return f"{mantissa.replace('.', ',')} × 10{sobrescrito}"


def plot_convergence(history, iterations):
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.semilogy(range(len(history)), history, marker="o", markersize=3,
                  linewidth=1.2, label="Custo")
    axis.set_xlabel("Iteração")
    axis.set_ylabel("Função de custo (escala log)")
    axis.set_title("Convergência do gradiente conjugado")
    axis.grid(True, alpha=0.25, which="both")
    axis.legend()
    axis.annotate(
        f"convergência na iteração {iterations}",
        xy=(iterations, history[-1]),
        xytext=(iterations * 0.42, history[0] * 1e-2),
        arrowprops={"arrowstyle": "->", "color": "tab:red",
                    "connectionstyle": "arc3,rad=-0.15"},
        color="tab:red",
        ha="right",
    )
    fig.savefig(OUTPUT_DIR / "convergencia_cg.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ratings_histogram(ratings_per_movie):
    fig, axis = plt.subplots(figsize=(9, 6))
    bins = np.logspace(0, np.log10(ratings_per_movie.max()), 30)
    axis.hist(ratings_per_movie, bins=bins, color="tab:blue", alpha=0.75)
    axis.axvline(MIN_RATINGS_FILTER, color="tab:red", linestyle="--",
                 label=f"filtro = {MIN_RATINGS_FILTER} avaliações")
    axis.set_xscale("log")
    axis.set_xlabel("Número de avaliações por filme (escala log)")
    axis.set_ylabel("Quantidade de filmes")
    axis.set_title("Distribuição de avaliações por filme")
    axis.grid(True, alpha=0.25, which="both")
    axis.legend()

    one = int(np.sum(ratings_per_movie == 1))
    menos_que_filtro = int(np.sum(ratings_per_movie < MIN_RATINGS_FILTER))
    total = len(ratings_per_movie)
    axis.annotate(
        f"{one} filmes ({br(100 * one / total, 1)}%) têm 1 avaliação",
        xy=(1.05, one * 0.92), xytext=(1.6, one * 0.78),
        color="tab:red",
        fontsize=13,
    )
    print(f"Filtro de {MIN_RATINGS_FILTER}: {menos_que_filtro} de {total} filmes "
          f"({br(100 * menos_que_filtro / total, 1)}%) ficam de fora")
    fig.savefig(OUTPUT_DIR / "histograma_avaliacoes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_top10(scores, ratings_per_movie, movie_names):
    ordem = np.argsort(scores)[::-1]
    cru = ordem[:TOP_N]
    elegiveis = [i for i in ordem if ratings_per_movie[i] >= MIN_RATINGS_FILTER]
    filtrado = elegiveis[:TOP_N]

    linhas = [
        "# Dez filmes de maior nota média predita",
        "",
        "Notas preditas pela fatoração com 100 atributos latentes e sem "
        "regularização. A coluna de avaliações indica quantas notas observadas "
        "o filme possui na base.",
        "",
        "## Lista completa",
        "",
        "| # | Filme | Nota média predita | Avaliações observadas |",
        "| --- | --- | --- | --- |",
    ]
    for pos, i in enumerate(cru, start=1):
        linhas.append(f"| {pos} | {movie_names[i]} | {br(scores[i])} | "
                      f"{int(ratings_per_movie[i])} |")

    linhas += [
        "",
        f"## Lista com filtro de {MIN_RATINGS_FILTER} avaliações",
        "",
        "| # | Filme | Nota média predita | Avaliações observadas |",
        "| --- | --- | --- | --- |",
    ]
    for pos, i in enumerate(filtrado, start=1):
        linhas.append(f"| {pos} | {movie_names[i]} | {br(scores[i])} | "
                      f"{int(ratings_per_movie[i])} |")

    (OUTPUT_DIR / "top10_filmes.md").write_text("\n".join(linhas) + "\n",
                                                encoding="utf-8")

    print("\nTop-10 completo (nome, nota, avaliações):")
    for pos, i in enumerate(cru, start=1):
        print(f"  {pos:2d}. {movie_names[i]} — {br(scores[i])} "
              f"({int(ratings_per_movie[i])} aval.)")
    print(f"\nTop-10 com filtro de {MIN_RATINGS_FILTER} avaliações:")
    for pos, i in enumerate(filtrado, start=1):
        print(f"  {pos:2d}. {movie_names[i]} — {br(scores[i])} "
              f"({int(ratings_per_movie[i])} aval.)")


def write_metrics(Y, R, predictions, history, iterations, n_params):
    observadas = R.astype(bool)
    residuos = predictions[observadas] - Y[observadas]
    rmse = float(np.sqrt(np.mean(residuos ** 2)))
    mae = float(np.mean(np.abs(residuos)))
    densidade = 100 * R.sum() / R.size

    texto = [
        "# Métricas da filtragem colaborativa",
        "",
        "| Métrica | Valor |",
        "| --- | --- |",
        f"| RMSE (avaliações observadas) | {cientifica(rmse)} |",
        f"| MAE (avaliações observadas) | {cientifica(mae)} |",
        f"| Custo final | {cientifica(history[-1], 3)} |",
        f"| Iterações até a convergência | {iterations} |",
        f"| Parâmetros ajustados | {n_params:,} |".replace(",", "."),
        f"| Densidade da matriz de avaliações | {br(densidade)}% |",
        "",
        "O RMSE e o MAE são calculados sobre as mesmas avaliações usadas no "
        "ajuste, portanto medem a capacidade de reconstruir as notas "
        "observadas, e não a generalização para avaliações não observadas.",
    ]
    (OUTPUT_DIR / "metricas.md").write_text("\n".join(texto) + "\n",
                                            encoding="utf-8")
    print(f"\nRMSE (in-sample): {cientifica(rmse)} | MAE: {cientifica(mae)}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    Y, R, movie_names = load_movie_data()
    history = []
    predictions, iterations, n_params = predict_ratings(Y, R, history)

    scores = predictions.mean(axis=1)
    ratings_per_movie = R.sum(axis=1)

    plot_convergence(history, iterations)
    plot_ratings_histogram(ratings_per_movie)
    write_top10(scores, ratings_per_movie, movie_names)
    write_metrics(Y, R, predictions, history, iterations, n_params)

    print(f"\nArtefatos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
