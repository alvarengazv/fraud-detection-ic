import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

CORES_PRINCIPAIS = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22"]
sns.set_theme(style="whitegrid", palette=CORES_PRINCIPAIS)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

OUTPUT_DIR = "./output/eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def salvar(fig, nome: str):
    caminho = os.path.join(OUTPUT_DIR, nome)
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> Dados salvos em: {caminho}")

def plot_tipos_atributos(df_t: pd.DataFrame, contagem: pd.Series):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(contagem.values, labels=contagem.index,
           autopct="%1.0f%%", colors=CORES_PRINCIPAIS[:len(contagem)],
           startangle=140)
    ax.set_title("Distribuição dos Tipos de Atributos\n(transactions.csv)")
    fig.tight_layout()
    salvar(fig, "tipos_atributos.png")

def plot_variavel_alvo(df_t: pd.DataFrame, vc: pd.Series, vc_pct: pd.Series, col_alvo: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cores_bar = [CORES_PRINCIPAIS[1] if i == 0 else CORES_PRINCIPAIS[2] for i in range(len(vc))]
    barras = ax.bar([str(x) for x in vc.index], vc.values, color=cores_bar)
    
    ax.set_title(f"Distribuição da Variável Alvo ({col_alvo})", fontsize=14, fontweight="bold", pad=20)
    ax.set_ylabel("Contagem Absoluta", fontsize=12)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, barra in enumerate(barras):
        valor_abs = vc.values[i]
        valor_pct = vc_pct.values[i]
        
        texto_rotulo = f"{valor_abs:,}\n({valor_pct:.2f}%)"
        
        ax.text(barra.get_x() + barra.get_width() / 2, 
                barra.get_height() + (vc.max() * 0.02), 
                texto_rotulo, 
                ha="center", 
                va="bottom", 
                fontsize=11, 
                fontweight="bold")

    ax.set_ylim(0, vc.max() * 1.15)

    fig.tight_layout()
    salvar(fig, "variavel_alvo.png")

def plot_tipos_fraude(df_t: pd.DataFrame, col_tipo: str, col_alvo: str):
    if col_tipo and col_tipo != col_alvo:
        fig2, ax = plt.subplots(figsize=(10, 4))
        vc_tipo = df_t[col_tipo].value_counts()
        ax.barh([str(x) for x in vc_tipo.index], vc_tipo.values,
                color=CORES_PRINCIPAIS[:len(vc_tipo)])
        ax.set_title("Contagem por Tipo de Transação/Fraude")
        ax.set_xlabel("Contagem")
        for i, v in enumerate(vc_tipo.values):
            ax.text(v + vc_tipo.max() * 0.005, i, f"{v:,}", va="center", fontsize=8)
        fig2.tight_layout()
        salvar(fig2, "tipos_fraude.png")

def plot_distribuicoes_numericas(df_t: pd.DataFrame, col_alvo: str, nrows_plot: int, ncols_plot: int, num_cols: list):
    fig, axes = plt.subplots(nrows_plot, ncols_plot, figsize=(5 * ncols_plot, 3.5 * nrows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        dados = df_t[col].dropna()
        axes[i].hist(dados, bins=50, color=CORES_PRINCIPAIS[2], alpha=0.7, density=True, edgecolor="white", linewidth=0.3)
        try:
            dados.plot.kde(ax=axes[i], color=CORES_PRINCIPAIS[0], linewidth=1.5)
        except Exception:
            pass
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")

        sk = dados.skew()
        sk_label = f"skew={sk:.2f}"
        axes[i].text(0.97, 0.95, sk_label, transform=axes[i].transAxes, ha="right", va="top", fontsize=7, color="gray")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuição dos Atributos Numéricos", fontsize=13, fontweight="bold")
    fig.tight_layout()
    salvar(fig, "distribuicoes_numericas.png")

def plot_boxplots_por_classe(df_t: pd.DataFrame, col_alvo: str, nrows_plot: int, ncols_plot: int, num_cols: list):
    if col_alvo and df_t[col_alvo].nunique() <= 15:
        fig2, axes2 = plt.subplots(nrows_plot, ncols_plot, figsize=(5 * ncols_plot, 3.5 * nrows_plot))
        axes2 = np.array(axes2).flatten()

        for i, col in enumerate(num_cols):
            grupos = [df_t.loc[df_t[col_alvo] == cl, col].dropna().values for cl in df_t[col_alvo].unique()]
            bp = axes2[i].boxplot(grupos, patch_artist=True, notch=False, medianprops={"color": "red", "linewidth": 1.5})
            for patch, cor in zip(bp["boxes"], CORES_PRINCIPAIS):
                patch.set_facecolor(cor)
                patch.set_alpha(0.6)
            axes2[i].set_title(f"{col} por {col_alvo}")
            axes2[i].set_xticklabels([str(x) for x in df_t[col_alvo].unique()], rotation=30)

        for j in range(i + 1, len(axes2)):
            axes2[j].set_visible(False)

        fig2.suptitle(f"Boxplots por Classe ({col_alvo})", fontsize=13, fontweight="bold")
        fig2.tight_layout()
        salvar(fig2, "boxplots_por_classe.png")

def plot_valores_ausentes(df_t: pd.DataFrame, df_p: pd.DataFrame, df_f: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(max(8, df_t.shape[1] * 0.5), 3))
    mascara_nulos = df_t.isna().astype(int)
    if mascara_nulos.sum().sum() > 0:
        amostra = mascara_nulos.sample(min(5000, len(mascara_nulos)), random_state=42)
        sns.heatmap(amostra.T, cbar=False, cmap=["#2ECC71", "#E74C3C"], xticklabels=False, ax=ax)
        ax.set_title("Mapa de Valores Ausentes — transactions.csv\n" "(verde=presente, vermelho=ausente)")
    else:
        ax.text(0.5, 0.5, "Nenhum valor ausente!", ha="center", va="center", fontsize=14, color="green", transform=ax.transAxes)
        ax.set_title("Mapa de Valores Ausentes — transactions.csv")
        ax.axis("off")

    fig.tight_layout()
    salvar(fig, "valores_ausentes.png")

def plot_outliers_por_atributo(df_t: pd.DataFrame, num_cols: list, df_out: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols) * 1.2), 4))
    cores_out = [CORES_PRINCIPAIS[1] if p > 5 else CORES_PRINCIPAIS[3] for p in df_out["%"]]
    ax.bar(df_out["Atributo"], df_out["%"], color=cores_out)
    ax.axhline(5, linestyle="--", color="orange", alpha=0.7, label="5% de referência")
    ax.set_title("% de Outliers por Atributo (Regra 3-sigma)")
    ax.set_ylabel("% de Outliers")
    ax.set_xlabel("Atributo")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    salvar(fig, "outliers_por_atributo.png")

def plot_volume_temporal(df_t: pd.DataFrame, col_data: str, datas: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 3))
    datas.dt.to_period("M").value_counts().sort_index().plot.bar(
        ax=ax, color=CORES_PRINCIPAIS[2])
    ax.set_title("Volume de Transações por Mês")
    ax.set_xlabel("Período")
    ax.set_ylabel("Contagem")
    plt.xticks(rotation=45)
    fig.tight_layout()
    salvar(fig, "volume_temporal.png")

def plot_correlacao_heatmap(num_cols: list, corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols) * 0.8), max(5, len(num_cols) * 0.7)))
    mascara = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask=mascara, ax=ax, linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Matriz de Correlação (Pearson) — Atributos Numéricos")
    fig.tight_layout()
    salvar(fig, "correlacao_heatmap.png")

def plot_correlacao_com_alvo(corr_alvo: pd.Series, col_alvo: str):
    fig2, ax2 = plt.subplots(figsize=(8, max(4, len(corr_alvo) * 0.4)))
    corr_alvo_plot = corr_alvo.drop(col_alvo, errors="ignore")
    cores_corr = [CORES_PRINCIPAIS[1] if v < 0 else CORES_PRINCIPAIS[2] for v in corr_alvo_plot.values]
    ax2.barh(corr_alvo_plot.index, corr_alvo_plot.values, color=cores_corr)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title(f"Correlação com '{col_alvo}'")
    ax2.set_xlabel("Pearson r")
    fig2.tight_layout()
    salvar(fig2, "correlacao_com_alvo.png")

def plot_valor_por_classe(df_t: pd.DataFrame, col_alvo: str, col_valor: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    try:
        classes = df_t[col_alvo].unique()
        dados_violin = [df_t.loc[df_t[col_alvo] == cl, col_valor].dropna().values for cl in classes]
        vp = axes[0].violinplot(dados_violin, showmedians=True)
        for i, pc in enumerate(vp["bodies"]):
            pc.set_facecolor(CORES_PRINCIPAIS[i % len(CORES_PRINCIPAIS)])
            pc.set_alpha(0.6)
        axes[0].set_xticks(range(1, len(classes) + 1))
        axes[0].set_xticklabels([str(c) for c in classes], rotation=30)
        axes[0].set_title(f"Violino: {col_valor} por {col_alvo}")
        axes[0].set_ylabel(col_valor)
    except Exception:
        axes[0].set_visible(False)
        
    for cl, cor in zip(df_t[col_alvo].unique(), CORES_PRINCIPAIS):
        subset = df_t.loc[df_t[col_alvo] == cl, col_valor].dropna()
        axes[1].hist(subset + 1, bins=60, alpha=0.5, label=str(cl), color=cor, density=True)
    axes[1].set_xscale("log")
    axes[1].set_title(f"Histograma (log) — {col_valor} por {col_alvo}")
    axes[1].set_xlabel(col_valor + " (escala log)")
    axes[1].set_ylabel("Densidade")
    axes[1].legend()

    fig.suptitle("Distribuição do Valor das Transações", fontsize=13, fontweight="bold")
    fig.tight_layout()
    salvar(fig, "valor_por_classe.png")

def plot_mediana_valor_por_tipo(df_t: pd.DataFrame, col_tipo: str, col_valor: str):
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    medias = df_t.groupby(col_tipo)[col_valor].median().sort_values(ascending=False)
    ax3.bar([str(x) for x in medias.index], medias.values, color=CORES_PRINCIPAIS[:len(medias)])
    ax3.set_title(f"Mediana de '{col_valor}' por Tipo")
    ax3.set_xlabel("Tipo")
    ax3.set_ylabel("Mediana")
    plt.xticks(rotation=30, ha="right")
    fig3.tight_layout()
    salvar(fig3, "mediana_valor_por_tipo.png")
