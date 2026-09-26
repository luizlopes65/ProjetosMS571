# ProjetosMS571

Projeto de uma rede neural para classificação de imagens, com suporte a diferentes arquiteturas, regularização e comparação entre otimizadores.

## Instalação

O projeto usa Python 3.12 ou superior. Com `uv`:

```bash
uv sync
```

## Executar o treinamento

```bash
python train.py
```

A arquitetura padrão usa 400 entradas, uma camada oculta com 25 neurônios e 10 classes.

Para testar outra arquitetura, use `train_model` com `hidden_layer_sizes`, por exemplo:

```python
train_model(X, y, hidden_layer_sizes=[25, 15])
```

## Gradient check

Verifica os gradientes do backpropagation comparando-os com diferenciação numérica:

```bash
python experiments/gradient_check.py
```

## Gerar figuras, tabelas e relatório

O script abaixo executa os experimentos principais e gera um relatório completo:

```bash
python experiments/generate_report.py
```

Os resultados são salvos em `results/`:

- `REPORT.md`: resumo dos experimentos e links para os arquivos gerados;
- `figures/`: gráficos e GIFs;
- `tables/`: tabelas em CSV e Markdown.

Para uma execução rápida:

```bash
python experiments/generate_report.py \
  --iterations 10 \
  --optimizer-iterations 5 10 \
  --output-dir results-smoke
```

A arquitetura e a seed também podem ser configuradas:

```bash
python experiments/generate_report.py \
  --hidden-layer-sizes 25 15 \
  --seed 42
```

## Principais experimentos

- busca do parâmetro de regularização `lambda`;
- curvas de aprendizado;
- evolução dos pesos dos neurônios;
- casos classificados incorretamente;
- comparação entre gradient descent e conjugate gradient;
- verificação numérica dos gradientes.
