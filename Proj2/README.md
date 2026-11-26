# Projeto 2 de Introdução a Machine Learning (MS571)

O projeto aplica PCA e K-means a imagens de faces e usa filtragem
colaborativa para estimar notas de filmes. Os dados de entrada estão em
`data/` e os resultados gerados estão em `results/`.

## Como executar

Pré-requisitos: Python 3.12 ou superior da série 3.x e uv.

Na raiz do repositório, entre em `Proj2` e instale as dependências fixadas em
`uv.lock`. Se estiver usando o ZIP, entre na pasta extraída `Proj2`.

```sh
cd Proj2
uv sync --locked
MPLBACKEND=Agg uv run --locked python pca_processing.py
MPLBACKEND=Agg uv run --locked python kmeans_clustering.py
uv run --locked python collaborative_filtering.py
MPLBACKEND=Agg uv run --locked python recommender_analysis.py
```

`MPLBACKEND=Agg` salva as figuras sem abrir janelas. Remova a variável para
exibir as figuras interativas de PCA e K-means.

## Scripts e resultados

| Script | O que faz | Saída |
| --- | --- | --- |
| `pca_processing.py` | Padroniza os pixels, calcula o PCA, exibe 36 eigenfaces e reconstrói 100 imagens com 100 componentes. | [Figuras de PCA](results/pca/) |
| `kmeans_clustering.py` | Agrupa até 1.000 faces nos 1.024 pixels originais, calcula a inércia para k de 1 a 8 e usa PCA para visualizar os quatro grupos escolhidos. | [Figuras de K-means](results/kmeans/) |
| `collaborative_filtering.py` | Centraliza as notas por filme, ajusta 100 atributos latentes por gradiente conjugado e ordena os filmes pela nota média predita. | Top 10 no terminal |
| `recommender_analysis.py` | Reutiliza o ajuste, registra a convergência e gera métricas, histograma e rankings com e sem o filtro de 20 avaliações. | [Relatórios do recomendador](results/recommender/) |

`data/face_data.mat` contém as faces de 32 × 32 pixels. Em
`data/movie_data.mat`, `Y` contém as notas e `R` indica as observadas.
`data/movie_names.txt` associa os nomes às linhas da matriz de notas.
`utils.py` reúne as operações numéricas e a montagem das grades de imagens.

Os valores de RMSE e MAE medem a reconstrução das notas observadas usadas
no ajuste. A execução regrava os resultados nas pastas indicadas.
