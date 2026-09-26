# Relatório de experimentos

## Configuração

- Dados: `/Users/luizlopes/Codebase/Personal/Unicamp2026+/IntroMachineLearning/ProjetosMS571/processed_data.csv`
- Seed: `31`
- Arquitetura: `[400, 25, 10]`
- Iterações principais: `800`
- Partições: treino `3800`, validação `200`, teste `1000`

## Gradient check

| Arquitetura | Resultado |
| --- | --- |
| `3-3` | OK |
| `3-5-3` | OK |
| `3-5-4-3` | OK |

## Busca de lambda

- Melhor lambda: `0`
- Acurácia de validação: `92.00%`
- Acurácia de teste: `93.10%`

## Curva de aprendizado

- Menor erro de treino: `0.00%`
- Menor erro de validação: `6.70%`

## Similaridade de cosseno entre classes

- Pares comparados como vetores de pixels achatados; a figura os reconstrói em 20×20.
- Maior similaridade interclasse: `0.9367` (classe 7 × classe 9).
- Este é um experimento descritivo dos dados; não treina uma CNN.

## Comparação entre otimizadores

- O grid abaixo usa apenas validação; o teste foi consultado uma vez para cada configuração selecionada.
- Melhor gradient descent: validação `92.00%`, teste `93.10%` (iterações=800, lambda=0)
- Melhor conjugate gradient: validação `95.00%`, teste `93.80%` (iterações=400, lambda=3)

## Arquivos gerados

### Figuras

- [Busca de lambda](figures/lambda_search_costs.png)
- [Erro de validação por lambda](figures/validation_error_vs_lambda.png)
- [Curva de aprendizado](figures/learning_curve.png)
- [Custo por tamanho de treino](figures/learning_curve_by_train_size.png)
- [Pares interclasse por similaridade de cosseno](figures/cosine_cross_class_pairs.png)
- [Evolução dos pesos](figures/activation_evolution.gif)
- [Casos classificados incorretamente](figures/error_cases.png)

### Tabelas

- [Comparação dos otimizadores em CSV](tables/optimizer_comparison.csv)
- [Comparação dos otimizadores em Markdown](tables/optimizer_comparison.md)
- [Resultados do gradient check](tables/gradient_check.md)
- [Pares interclasse por cosseno](tables/cosine_cross_class_pairs.csv)
