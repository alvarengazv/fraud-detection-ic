import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Configurações visuais ─────────────────────────────────────────────────────
CORES_PRINCIPAIS = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
                    "#9B59B6", "#1ABC9C", "#E67E22"]
sns.set_theme(style="whitegrid", palette=CORES_PRINCIPAIS)
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

OUTPUT_DIR = "./images/eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Utilitários ───────────────────────────────────────────────────────────────

def salvar(fig, nome: str):
    caminho = os.path.join(OUTPUT_DIR, nome)
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
    print(f"   [salvo] {caminho}")

def separador(titulo: str):

    print(f"---> Pressione Enter para continuar...")
    input()
    print("\n" + "=" * 70)
    print(f"  {titulo}")
    print("=" * 70)


def _classificar_atributo(serie: pd.Series) -> str:
    """Classifica o tipo do atributo conforme taxonomia do professor."""
    if serie.dtype == "object" or pd.api.types.is_categorical_dtype(serie):
        n_unicos = serie.nunique()
        if n_unicos == 2:
            return "Binário"
        return "Nominal"
    if pd.api.types.is_bool_dtype(serie):
        return "Binário"
    if pd.api.types.is_integer_dtype(serie):
        n_unicos = serie.nunique()
        if n_unicos == 2:
            return "Binário"
        if n_unicos <= 20:
            return "Discreto"
        return "Discreto"
    if pd.api.types.is_float_dtype(serie):
        return "Contínuo"
    return "Desconhecido"


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CONTEXTUALIZAÇÃO DO PROBLEMA
# ══════════════════════════════════════════════════════════════════════════════

def sec1_contextualizacao():
    separador("SEÇÃO 1 — CONTEXTUALIZAÇÃO DO PROBLEMA")

    print("""
  Problema: Detecção automática de transações financeiras fraudulentas.

  Contexto:
    Fraudes financeiras causam perdas bilionárias globalmente. Este dataset
    simula 1 milhão de transações bancárias, com 7 tipos distintos de fraude,
    permitindo treinar modelos de aprendizado de máquina para identificar
    padrões suspeitos antes que causem danos.

  Tarefa: CLASSIFICAÇÃO
    - Binária  → fraudulento (1) vs. legítimo (0)
    - Multiclasse → identificar o tipo específico de fraude

  Por que é um problema relevante?
    → Detecção em tempo real exige modelos rápidos e precisos.
    → Bases de fraude são tipicamente muito desbalanceadas (fraudes são raras).
    → Falsos negativos (fraude não detectada) têm custo muito maior que
      falsos positivos (transação legítima bloqueada).

  Fontes dos dados:
    Kaggle — sergionefedov/fraud-detection-1m-transactions-7-fraud-types
    """)


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — CARACTERIZAÇÃO GERAL DOS DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def sec2_caracterizacao(df_t: pd.DataFrame, df_p: pd.DataFrame, df_f: pd.DataFrame):
    separador("SEÇÃO 2 — CARACTERIZAÇÃO GERAL DOS DATASETS")

    for nome, df in [("transactions.csv", df_t), ("account_profiles.csv", df_p), ("fraud_patterns.csv", df_f)]:
        print(f"\n  ── {nome} ──────────────────────────────────────")
        print(f"  Instâncias : {df.shape[0]:>10,}")
        print(f"  Atributos  : {df.shape[1]:>10,}")
        print(f"  Memória    : {df.memory_usage(deep=True).sum() / 1e6:>10.2f} MB")

    # ── Taxonomia de atributos (slide do professor) ──────────────────────────
    print("\n\n  TAXONOMIA DOS ATRIBUTOS (transactions.csv)")
    print(f"  {'Atributo':<30} {'Dtype':<15} {'Tipo (professor)':<18} "
          f"{'#Únicos':>8} {'%Nulos':>8}")
    print("  " + "-" * 82)

    linhas = []
    for col in df_t.columns:
        tipo = _classificar_atributo(df_t[col])
        n_unicos = df_t[col].nunique()
        pct_nulo = df_t[col].isna().mean() * 100
        print(f"  {col:<30} {str(df_t[col].dtype):<15} {tipo:<18} "
              f"{n_unicos:>8,} {pct_nulo:>7.2f}%")
        linhas.append({"Atributo": col, "Tipo": tipo,
                       "Únicos": n_unicos, "%Nulos": pct_nulo})

    df_tax = pd.DataFrame(linhas)
    contagem_tipos = df_tax["Tipo"].value_counts()

    print("\n\n  Resumo por tipo de atributo:")
    for tipo, qtd in contagem_tipos.items():
        print(f"    {tipo:<18}: {qtd} atributo(s)")

    # Gráfico pizza dos tipos
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(contagem_tipos.values, labels=contagem_tipos.index,
           autopct="%1.0f%%", colors=CORES_PRINCIPAIS[:len(contagem_tipos)],
           startangle=140)
    ax.set_title("Distribuição dos Tipos de Atributos\n(transactions.csv)")
    fig.tight_layout()
    salvar(fig, "01_tipos_atributos.png")

    # Estatísticas descritivas
    print("\n\n  ESTATÍSTICAS DESCRITIVAS — Atributos numéricos:")
    desc = df_t.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["range"] = desc["max"] - desc["min"]
    desc["cv%"] = (desc["std"] / desc["mean"].abs() * 100).round(1)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_columns", 20, "display.width", 120):
        print(desc.to_string())

    return df_tax


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — ANÁLISE DA VARIÁVEL ALVO
# ══════════════════════════════════════════════════════════════════════════════

def sec3_variavel_alvo(df_t: pd.DataFrame):
    separador("SEÇÃO 3 — ANÁLISE DA VARIÁVEL ALVO")

    # Detectar coluna alvo automaticamente
    colunas_alvo_candidatas = [c for c in df_t.columns
                               if any(p in c.lower()
                                      for p in ["fraud", "label", "target",
                                                "is_fraud", "class", "tipo"])]
    if not colunas_alvo_candidatas:
        print("  [AVISO] Nenhuma coluna alvo identificada automaticamente.")
        print("  Colunas disponíveis:", df_t.columns.tolist())
        return None, None

    col_alvo = colunas_alvo_candidatas[0]
    print(f"\n  Coluna alvo identificada: '{col_alvo}'")

    vc = df_t[col_alvo].value_counts()
    vc_pct = df_t[col_alvo].value_counts(normalize=True) * 100

    print(f"\n  {'Classe':<30} {'Contagem':>10} {'%':>8}")
    print("  " + "-" * 52)
    for cls in vc.index:
        print(f"  {str(cls):<30} {vc[cls]:>10,} {vc_pct[cls]:>7.2f}%")

    # Calcular razão de desbalanceamento
    razao = vc.max() / vc.min()
    print(f"\n  Razão de desbalanceamento: {razao:.1f}:1")
    if razao > 10:
        print("  ⚠  DESBALANCEAMENTO SEVERO — estratégias necessárias: "
              "SMOTE, class_weight, under/oversampling.")
    elif razao > 3:
        print("  ⚠  Desbalanceamento moderado — monitorar métricas como F1, AUC-ROC.")

    # ── Gráficos ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Barras absolutas
    cores_bar = [CORES_PRINCIPAIS[1] if i == 0 else CORES_PRINCIPAIS[2]
                 for i in range(len(vc))]
    axes[0].bar([str(x) for x in vc.index], vc.values, color=cores_bar)
    axes[0].set_title(f"Distribuição Absoluta — {col_alvo}")
    axes[0].set_ylabel("Contagem")
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + vc.max() * 0.01, f"{v:,}", ha="center",
                     fontsize=8, fontweight="bold")

    # Barras percentuais
    axes[1].bar([str(x) for x in vc_pct.index], vc_pct.values, color=cores_bar)
    axes[1].set_title(f"Distribuição Percentual — {col_alvo}")
    axes[1].set_ylabel("%")
    axes[1].axhline(50, linestyle="--", color="gray", alpha=0.5, label="50%")
    for i, v in enumerate(vc_pct.values):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center",
                     fontsize=8, fontweight="bold")

    fig.suptitle("Distribuição da Variável Alvo", fontsize=14, fontweight="bold")
    fig.tight_layout()
    salvar(fig, "02_variavel_alvo.png")

    # Verificar coluna de tipo de fraude
    col_tipo = next((c for c in df_t.columns
                     if "type" in c.lower() or "kind" in c.lower()), None)
    if col_tipo and col_tipo != col_alvo:
        print(f"\n  Tipos de fraude ('{col_tipo}'):")
        print(df_t[col_tipo].value_counts().to_string())

        fig2, ax = plt.subplots(figsize=(10, 4))
        vc_tipo = df_t[col_tipo].value_counts()
        ax.barh([str(x) for x in vc_tipo.index], vc_tipo.values,
                color=CORES_PRINCIPAIS[:len(vc_tipo)])
        ax.set_title("Contagem por Tipo de Transação/Fraude")
        ax.set_xlabel("Contagem")
        for i, v in enumerate(vc_tipo.values):
            ax.text(v + vc_tipo.max() * 0.005, i, f"{v:,}", va="center", fontsize=8)
        fig2.tight_layout()
        salvar(fig2, "03_tipos_fraude.png")

    return col_alvo, col_tipo


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — DISTRIBUIÇÕES DOS ATRIBUTOS NUMÉRICOS
# ══════════════════════════════════════════════════════════════════════════════

def sec4_distribuicoes_numericas(df_t: pd.DataFrame, col_alvo: str):
    separador("SEÇÃO 4 — DISTRIBUIÇÕES DOS ATRIBUTOS NUMÉRICOS")

    num_cols = df_t.select_dtypes(include=np.number).columns.tolist()
    if col_alvo in num_cols and df_t[col_alvo].nunique() <= 10:
        num_cols = [c for c in num_cols if c != col_alvo]

    print(f"  Atributos numéricos analisados: {num_cols}")

    if not num_cols:
        print("  Nenhum atributo numérico encontrado.")
        return

    n = len(num_cols)
    ncols_plot = min(3, n)
    nrows_plot = (n + ncols_plot - 1) // ncols_plot

    # ── Histogramas + KDE ────────────────────────────────────────────────────
    fig, axes = plt.subplots(nrows_plot, ncols_plot,
                             figsize=(5 * ncols_plot, 3.5 * nrows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        dados = df_t[col].dropna()
        axes[i].hist(dados, bins=50, color=CORES_PRINCIPAIS[2],
                     alpha=0.7, density=True, edgecolor="white", linewidth=0.3)
        try:
            dados.plot.kde(ax=axes[i], color=CORES_PRINCIPAIS[0], linewidth=1.5)
        except Exception:
            pass
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("")

        # Assimetria e curtose no título
        sk = dados.skew()
        sk_label = f"skew={sk:.2f}"
        axes[i].text(0.97, 0.95, sk_label, transform=axes[i].transAxes,
                     ha="right", va="top", fontsize=7, color="gray")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuição dos Atributos Numéricos", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    salvar(fig, "04_distribuicoes_numericas.png")

    # ── Boxplots por classe alvo ─────────────────────────────────────────────
    if col_alvo and df_t[col_alvo].nunique() <= 15:
        fig2, axes2 = plt.subplots(nrows_plot, ncols_plot,
                                   figsize=(5 * ncols_plot, 3.5 * nrows_plot))
        axes2 = np.array(axes2).flatten()

        for i, col in enumerate(num_cols):
            grupos = [df_t.loc[df_t[col_alvo] == cl, col].dropna().values
                      for cl in df_t[col_alvo].unique()]
            bp = axes2[i].boxplot(grupos, patch_artist=True, notch=False,
                                  medianprops={"color": "red", "linewidth": 1.5})
            for patch, cor in zip(bp["boxes"], CORES_PRINCIPAIS):
                patch.set_facecolor(cor)
                patch.set_alpha(0.6)
            axes2[i].set_title(f"{col} por {col_alvo}")
            axes2[i].set_xticklabels([str(x) for x in df_t[col_alvo].unique()],
                                     rotation=30)

        for j in range(i + 1, len(axes2)):
            axes2[j].set_visible(False)

        fig2.suptitle(f"Boxplots por Classe ({col_alvo})", fontsize=13,
                      fontweight="bold")
        fig2.tight_layout()
        salvar(fig2, "05_boxplots_por_classe.png")

    # ── Resumo de assimetria ─────────────────────────────────────────────────
    print("\n  Assimetria (skewness) dos atributos numéricos:")
    print(f"  {'Atributo':<30} {'Skewness':>12} {'Interpretação':<20}")
    print("  " + "-" * 65)
    for col in num_cols:
        sk = df_t[col].skew()
        interp = ("simétrico" if abs(sk) < 0.5
                  else "assimétrico moderado" if abs(sk) < 1
                  else "assimétrico forte")
        print(f"  {col:<30} {sk:>12.4f}  {interp}")


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — ATRIBUTOS CATEGÓRICOS
# ══════════════════════════════════════════════════════════════════════════════

def sec5_atributos_categoricos(df_t: pd.DataFrame, col_alvo: str):
    separador("SEÇÃO 5 — ATRIBUTOS CATEGÓRICOS")

    cat_cols = df_t.select_dtypes(include=["object", "category"]).columns.tolist()
    if col_alvo in cat_cols:
        cat_cols = [c for c in cat_cols if c != col_alvo]

    if not cat_cols:
        print("  Nenhum atributo categórico encontrado além da variável alvo.")
        return

    print(f"  Atributos categóricos: {cat_cols}\n")

    n = len(cat_cols)
    ncols_plot = min(2, n)
    nrows_plot = (n + ncols_plot - 1) // ncols_plot

    fig, axes = plt.subplots(nrows_plot, ncols_plot,
                             figsize=(7 * ncols_plot, 4 * nrows_plot))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cat_cols):
        vc = df_t[col].value_counts().head(15)
        axes[i].barh([str(x) for x in vc.index], vc.values,
                     color=CORES_PRINCIPAIS[:len(vc)])
        axes[i].set_title(f"{col} (top {min(15, len(vc))} valores)")
        axes[i].set_xlabel("Contagem")
        axes[i].invert_yaxis()

        print(f"  '{col}': {df_t[col].nunique()} valores únicos")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuição dos Atributos Categóricos", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    salvar(fig, "06_atributos_categoricos.png")


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — QUALIDADE DOS DADOS
# ══════════════════════════════════════════════════════════════════════════════

def sec6_qualidade_dados(df_t: pd.DataFrame, df_p: pd.DataFrame, df_f: pd.DataFrame):
    separador("SEÇÃO 6 — QUALIDADE DOS DADOS")

    # 6.1 Valores ausentes ────────────────────────────────────────────────────
    print("\n  6.1 — VALORES AUSENTES")

    for nome, df in [("transactions.csv", df_t), ("account_profiles.csv", df_p), ("fraud_patterns.csv", df_f)]: 
        nulos = df.isna().sum()
        pct_nulos = df.isna().mean() * 100
        df_nulos = pd.DataFrame({"Ausentes": nulos, "%": pct_nulos}
                                ).query("Ausentes > 0").sort_values("%", ascending=False)

        if df_nulos.empty:
            print(f"\n  {nome}: ✓ Nenhum valor ausente encontrado.")
        else:
            print(f"\n  {nome}: {len(df_nulos)} colunas com valores ausentes")
            print(df_nulos.to_string())

    # Heatmap de nulos para transactions
    fig, ax = plt.subplots(figsize=(max(8, df_t.shape[1] * 0.5), 3))
    mascara_nulos = df_t.isna().astype(int)
    if mascara_nulos.sum().sum() > 0:
        # Amostrar para visualização
        amostra = mascara_nulos.sample(min(5000, len(mascara_nulos)),
                                        random_state=42)
        sns.heatmap(amostra.T, cbar=False, cmap=["#2ECC71", "#E74C3C"],
                    xticklabels=False, ax=ax)
        ax.set_title("Mapa de Valores Ausentes — transactions.csv\n"
                     "(verde=presente, vermelho=ausente)")
    else:
        ax.text(0.5, 0.5, "Nenhum valor ausente!", ha="center", va="center",
                fontsize=14, color="green", transform=ax.transAxes)
        ax.set_title("Mapa de Valores Ausentes — transactions.csv")
        ax.axis("off")

    fig.tight_layout()
    salvar(fig, "07_valores_ausentes.png")

    # 6.2 Duplicatas ──────────────────────────────────────────────────────────
    print("\n  6.2 — DUPLICATAS")
    dup_t = df_t.duplicated().sum()
    dup_p = df_p.duplicated().sum()
    print(f"  transactions.csv    : {dup_t:,} linhas duplicadas "
          f"({dup_t/len(df_t)*100:.3f}%)")
    print(f"  account_profiles.csv: {dup_p:,} linhas duplicadas "
          f"({dup_p/len(df_p)*100:.3f}%)")

    # 6.3 Outliers ────────────────────────────────────────────────────────────
    print("\n  6.3 — DETECÇÃO DE OUTLIERS (Regra 3-sigma, slides do professor)")
    print(f"\n  {'Atributo':<30} {'Outliers':>10} {'%':>8} {'Min':>12} "
          f"{'Max':>12} {'Média':>12} {'Std':>12}")
    print("  " + "-" * 100)

    num_cols = df_t.select_dtypes(include=np.number).columns
    dados_outliers = []

    for col in num_cols:
        serie = df_t[col].dropna()
        mu = serie.mean()
        sigma = serie.std()
        outliers = ((serie < mu - 3 * sigma) | (serie > mu + 3 * sigma)).sum()
        pct = outliers / len(serie) * 100
        dados_outliers.append({
            "Atributo": col, "Outliers": outliers, "%": pct,
            "Min": serie.min(), "Max": serie.max(),
            "Média": mu, "Std": sigma
        })
        print(f"  {col:<30} {outliers:>10,} {pct:>7.2f}%  "
              f"{serie.min():>12.2f} {serie.max():>12.2f} "
              f"{mu:>12.2f} {sigma:>12.2f}")

    df_out = pd.DataFrame(dados_outliers).sort_values("%", ascending=False)

    # Gráfico de barras de outliers
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols) * 1.2), 4))
    cores_out = [CORES_PRINCIPAIS[1] if p > 5 else CORES_PRINCIPAIS[3]
                 for p in df_out["%"]]
    ax.bar(df_out["Atributo"], df_out["%"], color=cores_out)
    ax.axhline(5, linestyle="--", color="orange", alpha=0.7, label="5% de referência")
    ax.set_title("% de Outliers por Atributo (Regra 3-sigma)")
    ax.set_ylabel("% de Outliers")
    ax.set_xlabel("Atributo")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    salvar(fig, "08_outliers_por_atributo.png")

    # 6.4 Consistência temporal (se houver data/hora) ─────────────────────────
    col_data = next((c for c in df_t.columns
                     if any(p in c.lower()
                            for p in ["date", "time", "timestamp", "data", "hora"])),
                    None)
    if col_data:
        print(f"\n  6.4 — VERIFICAÇÃO TEMPORAL ('{col_data}')")
        try:
            datas = pd.to_datetime(df_t[col_data], errors="coerce")
            print(f"  Data mínima : {datas.min()}")
            print(f"  Data máxima : {datas.max()}")
            print(f"  Datas nulas : {datas.isna().sum():,}")

            fig, ax = plt.subplots(figsize=(12, 3))
            datas.dt.to_period("M").value_counts().sort_index().plot.bar(
                ax=ax, color=CORES_PRINCIPAIS[2])
            ax.set_title("Volume de Transações por Mês")
            ax.set_xlabel("Período")
            ax.set_ylabel("Contagem")
            plt.xticks(rotation=45)
            fig.tight_layout()
            salvar(fig, "09_volume_temporal.png")
        except Exception as e:
            print(f"  [AVISO] Não foi possível processar coluna de data: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — CORRELAÇÕES E RELAÇÕES ENTRE ATRIBUTOS
# ══════════════════════════════════════════════════════════════════════════════

def sec7_correlacoes(df_t: pd.DataFrame, col_alvo: str):
    separador("SEÇÃO 7 — CORRELAÇÕES ENTRE ATRIBUTOS")

    num_cols = df_t.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) < 2:
        print("  Atributos numéricos insuficientes para análise de correlação.")
        return

    corr = df_t[num_cols].corr()

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols) * 0.8),
                                    max(5, len(num_cols) * 0.7)))
    mascara = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mascara, ax=ax, linewidths=0.5,
                annot_kws={"size": 8})
    ax.set_title("Matriz de Correlação (Pearson) — Atributos Numéricos")
    fig.tight_layout()
    salvar(fig, "10_correlacao_heatmap.png")

    # Atributos altamente correlacionados (candidatos à redundância)
    print("\n  Pares com |correlação| > 0.8 (possíveis redundâncias):")
    alto_corr = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            val = abs(corr.iloc[i, j])
            if val > 0.8:
                alto_corr.append((num_cols[i], num_cols[j], corr.iloc[i, j]))
                print(f"    {num_cols[i]:<25} ↔ {num_cols[j]:<25} : {corr.iloc[i, j]:+.4f}")

    if not alto_corr:
        print("  Nenhum par com |correlação| > 0.8 encontrado.")

    # Correlação com a variável alvo
    if col_alvo and col_alvo in df_t.columns:
        try:
            corr_alvo = df_t[num_cols].corrwith(
                pd.to_numeric(df_t[col_alvo], errors="coerce")
            ).sort_values(key=abs, ascending=False)

            print(f"\n  Correlação dos atributos numéricos com '{col_alvo}':")
            print(f"  {'Atributo':<30} {'Correlação':>12}")
            print("  " + "-" * 44)
            for attr, val in corr_alvo.items():
                if attr != col_alvo:
                    print(f"  {attr:<30} {val:>12.4f}")

            fig2, ax2 = plt.subplots(figsize=(8, max(4, len(corr_alvo) * 0.4)))
            corr_alvo_plot = corr_alvo.drop(col_alvo, errors="ignore")
            cores_corr = [CORES_PRINCIPAIS[1] if v < 0 else CORES_PRINCIPAIS[2]
                          for v in corr_alvo_plot.values]
            ax2.barh(corr_alvo_plot.index, corr_alvo_plot.values, color=cores_corr)
            ax2.axvline(0, color="black", linewidth=0.8)
            ax2.set_title(f"Correlação com '{col_alvo}'")
            ax2.set_xlabel("Pearson r")
            fig2.tight_layout()
            salvar(fig2, "11_correlacao_com_alvo.png")
        except Exception as e:
            print(f"  [AVISO] Não foi possível calcular correlação com alvo: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — ANÁLISE DO VALOR MONETÁRIO POR TIPO DE FRAUDE
# ══════════════════════════════════════════════════════════════════════════════

def sec8_analise_monetaria(df_t: pd.DataFrame, col_alvo: str, col_tipo: str):
    separador("SEÇÃO 8 — ANÁLISE DO VALOR DAS TRANSAÇÕES")

    col_valor = next((c for c in df_t.columns
                      if any(p in c.lower()
                             for p in ["amount", "valor", "value", "montant"])), None)

    if col_valor is None:
        print("  Coluna de valor monetário não identificada.")
        return

    print(f"  Coluna de valor: '{col_valor}'\n")

    # Estatísticas gerais
    desc = df_t[col_valor].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.99])
    print(desc.to_string())

    # Por classe
    if col_alvo and col_alvo in df_t.columns:
        print(f"\n  Estatísticas de '{col_valor}' por '{col_alvo}':")
        print(df_t.groupby(col_alvo)[col_valor].describe().to_string())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Violino
        try:
            classes = df_t[col_alvo].unique()
            dados_violin = [df_t.loc[df_t[col_alvo] == cl, col_valor].dropna().values
                            for cl in classes]
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

        # Histograma sobreposto (escala log)
        for cl, cor in zip(df_t[col_alvo].unique(), CORES_PRINCIPAIS):
            subset = df_t.loc[df_t[col_alvo] == cl, col_valor].dropna()
            axes[1].hist(subset + 1, bins=60, alpha=0.5, label=str(cl),
                         color=cor, density=True)
        axes[1].set_xscale("log")
        axes[1].set_title(f"Histograma (log) — {col_valor} por {col_alvo}")
        axes[1].set_xlabel(col_valor + " (escala log)")
        axes[1].set_ylabel("Densidade")
        axes[1].legend()

        fig.suptitle("Distribuição do Valor das Transações", fontsize=13,
                     fontweight="bold")
        fig.tight_layout()
        salvar(fig, "12_valor_por_classe.png")

    # Por tipo de fraude
    if col_tipo and col_tipo in df_t.columns and col_tipo != col_alvo:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        medias = df_t.groupby(col_tipo)[col_valor].median().sort_values(ascending=False)
        ax3.bar([str(x) for x in medias.index], medias.values,
                color=CORES_PRINCIPAIS[:len(medias)])
        ax3.set_title(f"Mediana de '{col_valor}' por Tipo")
        ax3.set_xlabel("Tipo")
        ax3.set_ylabel("Mediana")
        plt.xticks(rotation=30, ha="right")
        fig3.tight_layout()
        salvar(fig3, "13_mediana_valor_por_tipo.png")


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — MERGE E ANÁLISE CONJUNTA COM PERFIS DE CONTAS
# ══════════════════════════════════════════════════════════════════════════════

def sec9_analise_conjunta(df_t: pd.DataFrame, df_p: pd.DataFrame, df_f: pd.DataFrame, col_alvo: str):
    separador("SEÇÃO 9 — ANÁLISE CONJUNTA: TRANSAÇÕES + PERFIS DE CONTA")

    # Identificar chave de junção
    cols_t = set(df_t.columns)
    cols_p = set(df_p.columns)
    chaves_candidatas = cols_t & cols_p
    chaves_candidatas = [c for c in chaves_candidatas if c != col_alvo]

    print(f"  Colunas comuns (candidatas à chave): {list(chaves_candidatas)}")

    if not chaves_candidatas:
        print("  Sem colunas em comum para realizar merge.")
        return None

    chave = chaves_candidatas[0]
    print(f"  Usando '{chave}' como chave de junção.\n")

    df_merged = df_t.merge(df_p, on=chave, how="left", suffixes=("", "_perfil"))
    print(f"  Shape após merge: {df_merged.shape}")
    print(f"  Novos nulos introduzidos após merge:")

    nulos_novos = df_merged.isna().sum()
    nulos_novos = nulos_novos[nulos_novos > 0]
    if nulos_novos.empty:
        print("  Nenhum novo nulo introduzido — bom sinal para a qualidade do merge!")
    else:
        print(nulos_novos.to_string())

    return df_merged


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 10 — SÍNTESE E DECISÕES DE PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def sec10_sintese_preprocessamento(df_t: pd.DataFrame, col_alvo: str):
    separador("SEÇÃO 10 — SÍNTESE E PLANO DE PRÉ-PROCESSAMENTO")

    num_cols = df_t.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_t.select_dtypes(include=["object", "category"]).columns.tolist()
    nulos_total = df_t.isna().sum().sum()
    dups = df_t.duplicated().sum()

    # Outliers
    outliers_info = {}
    for col in num_cols:
        s = df_t[col].dropna()
        mu, sigma = s.mean(), s.std()
        out = ((s < mu - 3 * sigma) | (s > mu + 3 * sigma)).sum()
        outliers_info[col] = out / len(s) * 100

    has_outliers = any(v > 1 for v in outliers_info.values())

    # Desbalanceamento
    desbalanceado = False
    razao = 1.0
    if col_alvo and col_alvo in df_t.columns:
        vc = df_t[col_alvo].value_counts()
        if len(vc) >= 2:
            razao = vc.max() / vc.min()
            desbalanceado = razao > 3

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │              SÍNTESE DOS PROBLEMAS IDENTIFICADOS                │
  └─────────────────────────────────────────────────────────────────┘
""")
    problemas = [
        ("Valores ausentes",
         f"{nulos_total:,} nulos totais",
         "✓ Presente" if nulos_total > 0 else "✗ Ausente"),
        ("Dados duplicados",
         f"{dups:,} duplicatas",
         "✓ Presente" if dups > 0 else "✗ Ausente"),
        ("Outliers",
         f"Colunas afetadas: {sum(1 for v in outliers_info.values() if v > 1)}",
         "✓ Presente" if has_outliers else "✗ Ausente"),
        ("Desbalanceamento",
         f"Razão {razao:.1f}:1",
         "✓ Presente" if desbalanceado else "✗ Ausente"),
        ("Atributos categóricos",
         f"{len(cat_cols)} colunas",
         "✓ Presente" if cat_cols else "✗ Ausente"),
    ]

    for prob, detalhe, status in problemas:
        print(f"  {status:<15} {prob:<30} {detalhe}")

    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │              PLANO DE PRÉ-PROCESSAMENTO (próxima fase)          │
  └─────────────────────────────────────────────────────────────────┘

  1. VALORES AUSENTES
     → Se MCAR/MAR: imputação pela mediana (numérico) ou moda (categórico)
     → Se MNAR: avaliação caso a caso; possível remoção da variável

  2. OUTLIERS
     → Analisar contexto: outliers de fraude NÃO devem ser removidos!
     → Para atributos com outliers irrelevantes: capping (percentil 1%–99%)
     → Usar normalização z-score robusta (mediana/IQR) em vez de min-max

  3. CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS
     → Nominal (sem ordem): One-Hot Encoding
     → Ordinal (com ordem): Label Encoding / Ordinal Encoding
     → Alta cardinalidade: Target Encoding ou frequency encoding

  4. NORMALIZAÇÃO / PADRONIZAÇÃO
     → Algoritmos baseados em distância (KNN, SVM, RNA):
       usar z-score (média=0, std=1) — mais robusto a outliers
     → Algoritmos baseados em árvore (Random Forest, XGBoost):
       normalização NÃO é necessária

  5. DESBALANCEAMENTO
     → Estratégia 1: SMOTE (oversampling da classe minoritária)
     → Estratégia 2: class_weight='balanced' nos modelos
     → Estratégia 3: threshold tuning (ajuste do limiar de decisão)
     → Métricas: F1-Score, AUC-ROC, G-Mean (NÃO usar só acurácia)

  6. SELEÇÃO/REDUÇÃO DE ATRIBUTOS
     → Remover atributos com correlação > 0.95 (redundância)
     → Avaliar PCA se dimensionalidade for alta
     → Feature importance via Random Forest na fase de modelagem
""")


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def executar_eda(df_transactions: pd.DataFrame, df_account_profiles: pd.DataFrame, df_fraud_patterns: pd.DataFrame):

    print(f"\n  Gráficos serão salvos em: {os.path.abspath(OUTPUT_DIR)}/\n")

    # ── Seção 1: Contextualização ─────────────────────────────────────────────
    sec1_contextualizacao()

    # ── Seção 2: Caracterização geral ────────────────────────────────────────
    sec2_caracterizacao(df_transactions, df_account_profiles, df_fraud_patterns)

    # ── Seção 3: Variável alvo ────────────────────────────────────────────────
    col_alvo, col_tipo = sec3_variavel_alvo(df_transactions)

    # ── Seção 4: Distribuições numéricas ─────────────────────────────────────
    sec4_distribuicoes_numericas(df_transactions, col_alvo)

    # ── Seção 5: Atributos categóricos ───────────────────────────────────────
    sec5_atributos_categoricos(df_transactions, col_alvo)

    # ── Seção 6: Qualidade dos dados ─────────────────────────────────────────
    sec6_qualidade_dados(df_transactions, df_account_profiles, df_fraud_patterns)

    # ── Seção 7: Correlações ─────────────────────────────────────────────────
    sec7_correlacoes(df_transactions, col_alvo)

    # ── Seção 8: Análise monetária ───────────────────────────────────────────
    sec8_analise_monetaria(df_transactions, col_alvo, col_tipo)

    # ── Seção 9: Análise conjunta ─────────────────────────────────────────────
    df_merged = sec9_analise_conjunta(df_transactions, df_account_profiles, df_fraud_patterns, col_alvo)

    # ── Seção 10: Síntese e plano de pré-processamento ───────────────────────
    sec10_sintese_preprocessamento(df_transactions, col_alvo)

    separador("EDA CONCLUÍDA")
    print(f"\n  Todos os gráficos foram salvos em: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  Total de figuras geradas: "
          f"{len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])}\n")

    return df_merged