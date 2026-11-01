# Experimentos e artefatos

## Pipeline

```text
data/images.csv + data/labels.csv
              │ preprocess_data.py
              ▼
      processed_data.csv
              │ train.py / experiments
              ▼
  métricas, tabelas e figuras em results/<subdiretório>
```

As imagens possuem 400 atributos (20 × 20 pixels) e os rótulos são classes de
1 a 10. O particionamento padrão, com seed 42, reserva 76% para treino, 4%
para validação e 20% para teste.

## Experimentos incluídos

| Script ou rotina | Objetivo | Artefato principal |
| --- | --- | --- |
| `experiments/gradient_check.py` | Confere os gradientes analíticos por diferenças finitas. | saída do terminal e `tables/gradient_check.md` no relatório |
| `experiments/lambda_search.py` | Escolhe λ pela acurácia de validação. | `figures/lambda_search_costs.png` e `validation_error_vs_lambda.png` |
| `experiments/learning_curve.py` | Mede erro de treino e validação em diferentes tamanhos de amostra. | `figures/learning_curve*.png` |
| `experiments/error_cases_viz.py` | Exibe erros do modelo escolhido por validação. | `figures/error_cases.png` |
| `experiments/cosine_similarity_cases.py` | Encontra pares de imagens de classes distintas com alta similaridade. | `figures/cosine_cross_class_pairs.png` e tabela CSV |
| `experiments/conjugate_gradients.py` | Compara gradient descent e conjugate gradient. | `tables/optimizer_comparison.*` |
| `experiments/optimizer_comparison_graphs.py` | Converte a tabela de otimizadores em gráfico. | `figures/optimizer_comparison.png` |

`experiments/generate_report.py` é o ponto de entrada recomendado: executa
todas as rotinas com uma configuração comum e escreve um índice em `REPORT.md`.
Ele seleciona λ e configurações de otimizador usando apenas validação; o teste
é consultado uma vez para a configuração selecionada.

## Parâmetros comuns

- `--iterations`: número de atualizações do gradient descent. O padrão é 800
  para o treino e o relatório completo.
- `--optimizer-iterations`: lista de orçamentos avaliados para cada
  otimizador no relatório.
- `--hidden-layer-sizes`: quantos neurônios usar em cada camada escondida;
  por padrão, `25`.
- `--seed`: controla a partição e as inicializações aleatórias.
- `--output-dir`: diretório de saída do relatório. Prefira um diretório novo,
  como `results-local`, para não sobrescrever artefatos de referência.

## Critérios de verificação

Uma execução está saudável quando:

1. `preprocess_data.py` informa 5.000 exemplos e 400 atributos.
2. `gradient_check.py` termina com `TODOS OS TESTES PASSARAM`.
3. `uv run python -m unittest discover -s tests -v` termina sem falhas.
4. O diretório de saída do relatório contém `REPORT.md`, `figures/` e
   `tables/`.
