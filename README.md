# Projetos MS571 — rede neural para imagens

Implementação didática, em NumPy, de uma rede neural multicamada para
classificar imagens 20 × 20. O projeto inclui pré-processamento, treino com
gradient descent, busca de regularização, gradient checking, comparação com
conjugate gradient e geração de figuras.

## Executar do zero

Os comandos abaixo devem ser executados na raiz do repositório. É necessário
ter [Python 3.12](https://www.python.org/downloads/) e
[uv](https://docs.astral.sh/uv/) instalados.

```bash
git clone https://github.com/luizlopes65/ProjetosMS571.git
cd ProjetosMS571
uv sync
uv run python preprocess_data.py
uv run python train.py --iterations 10
```

O último comando é uma verificação rápida do fluxo completo. Ele mostra o
custo final e a acurácia no conjunto de teste. Para o treinamento padrão de
800 iterações, remova `--iterations 10`:

```bash
uv run python train.py
```

`preprocess_data.py` lê `data/images.csv` e `data/labels.csv`, converte os
valores com vírgula decimal e cria `processed_data.csv`. O arquivo processado
já acompanha o repositório, mas executar a etapa torna a origem dos dados e a
reprodução explícitas.

## Verificar a implementação

O gradient checking compara o backpropagation com diferenças finitas em redes
pequenas. Todos os casos devem terminar com `TODOS OS TESTES PASSARAM`.

```bash
uv run python experiments/gradient_check.py
uv run python -m unittest discover -s tests -v
```

## Gerar os experimentos

Para uma execução de fumaça que gera relatório, tabelas e figuras sem alterar
os artefatos de referência em `results/`, use um diretório local:

```bash
uv run python experiments/generate_report.py \
  --iterations 1 \
  --optimizer-iterations 1 \
  --output-dir results-smoke
```

O relatório é criado em `results-smoke/REPORT.md`. Mesmo esta execução roda
todas as análises; ela é destinada a validar o fluxo, não a medir desempenho.
Para gerar os resultados completos, execute (a duração depende da máquina):

```bash
uv run python experiments/generate_report.py --output-dir results-local
```

Os resultados completos ficam em `results-local/`:

- `REPORT.md`: configuração, métricas e links para os artefatos;
- `figures/`: busca de λ, curva de aprendizado, erros, pesos e otimizadores;
- `tables/`: gradient check, pares de cosseno e comparação de otimizadores.

## Comandos úteis

```bash
# Usar somente os primeiros 100 exemplos durante uma inspeção rápida.
uv run python preprocess_data.py --limit 100 --output /tmp/dataset.csv

# Alterar arquitetura, regularização e seed do treino.
uv run python train.py \
  --hidden-layer-sizes 25 15 \
  --lambda 0.1 \
  --seed 42

# Recriar somente o gráfico de comparação a partir de uma tabela existente.
uv run python experiments/optimizer_comparison_graphs.py \
  --input results-local/tables/optimizer_comparison.csv \
  --output results-local/figures/optimizer_comparison.png
```

Consulte [a documentação dos experimentos](docs/EXPERIMENTS.md) para a
descrição de cada análise, entradas e arquivos produzidos.

## Estrutura do projeto

```text
data/               dados brutos: imagens e rótulos
docs/               documentação de execução e experimentos
experiments/        análises reproduzíveis e geração de relatório
results/            artefatos de referência já gerados
tests/              testes automatizados do fluxo principal
preprocess_data.py  conversão dos CSVs brutos
train.py            particionamento, treino e avaliação
utils.py            forward pass, custo, backpropagation e gradient descent
```

## Reprodutibilidade

`uv.lock` fixa as versões das dependências. Os comandos usam `uv run` para
garantir que o Python e as bibliotecas do ambiente do projeto sejam usados.
As sementes padrão tornam as partições e as inicializações reproduzíveis.
