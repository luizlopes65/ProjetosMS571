from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Definir os caminhos corretos com base na estrutura de pastas
# __file__ está em: ProjetosMS571/experiments/optimizer_comparison_graphs.py
SCRIPT_DIR = Path(__file__).resolve().parent      # Pasta experiments/
ROOT_DIR = SCRIPT_DIR.parent                      # Pasta raiz ProjetosMS571/

# Caminho para o CSV: ProjetosMS571/results/tables/optimizer_comparison.csv
CSV_PATH = ROOT_DIR / 'results' / 'tables' / 'optimizer_comparison.csv'

# Pasta onde a figura gerada será salva: ProjetosMS571/results/figures/
FIGURES_DIR = ROOT_DIR / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)  # Cria a pasta caso não exista

# 2. Carregar os dados
df = pd.read_csv(CSV_PATH)
lambdas = df['lambda'].unique()

# 3. Criar os gráficos lado a lado com eixos compartilhados
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# --- Gráfico 1: Gradiente Descendente ---
for lmbda in lambdas:
    sub_df = df[df['lambda'] == lmbda].sort_values('iterations')
    axes[0].plot(
        sub_df['iterations'], 
        sub_df['gradient_descent_validation_accuracy'], 
        marker='o', 
        label=f'λ = {lmbda}'
    )

axes[0].set_title('Gradiente Descendente', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Número de Iterações', fontsize=12)
axes[0].set_ylabel('Acurácia de Validação (%)', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(loc='lower right', fontsize=10, framealpha=0.8)

# --- Gráfico 2: Gradiente Conjugado ---
for lmbda in lambdas:
    sub_df = df[df['lambda'] == lmbda].sort_values('iterations')
    axes[1].plot(
        sub_df['iterations'], 
        sub_df['conjugate_gradient_validation_accuracy'], 
        marker='s', 
        label=f'λ = {lmbda}'
    )

axes[1].set_title('Gradiente Conjugado', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Número de Iterações', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(loc='lower right', fontsize=10, framealpha=0.8)

plt.tight_layout()

# Salvar o gráfico diretamente na pasta results/figures/ do seu projeto
plt.savefig(FIGURES_DIR / 'optimizer_comparison.png', dpi=300, bbox_inches='tight')
plt.show()