import pandas as pd
import numpy as np
import os
import eda.plots as plots

COLUNAS_ID = {"transaction_id", "account_id", "timestamp"}

def separador(titulo: str):
    print(f"\n  -- {titulo} ---------------")
    
def classificar_atributo(serie: pd.Series) -> str:
    serie = serie.dropna()
    nome_coluna = str(serie.name).lower() if serie.name else ""
    n_unicos = serie.nunique()

    if pd.api.types.is_bool_dtype(serie) or n_unicos == 2:
        return "Binário"
        
    if (
        pd.api.types.is_object_dtype(serie) 
        or pd.api.types.is_string_dtype(serie) 
        or isinstance(serie.dtype, pd.CategoricalDtype)
    ):
        if isinstance(serie.dtype, pd.CategoricalDtype) and serie.dtype.ordered:
            return "Ordinal"
        return "Nominal"

    if pd.api.types.is_numeric_dtype(serie):
        if serie.min() < 0:
            return "Intervalo"
        if pd.api.types.is_integer_dtype(serie):
            return "Discreto"
            
        if pd.api.types.is_float_dtype(serie):
            return "Contínuo"

    return "Desconhecido"

def caracterizacao(df_t: pd.DataFrame, df_p: pd.DataFrame, df_f: pd.DataFrame):
    separador("CARACTERIZAÇÃO GERAL DOS DATASETS")

    for nome, df in [("transactions.csv", df_t), ("account_profiles.csv", df_p), ("fraud_patterns.csv", df_f)]:
        print(f"\n  --> {nome}")
        print(f"  Instâncias : {df.shape[0]:>10,}")
        print(f"  Atributos  : {df.shape[1]:>10,}")
        print(f"  Memória    : {df.memory_usage(deep=True).sum() / 1e6:>10.2f} MB")

    print("\n\n  --> TAXONOMIA DOS ATRIBUTOS — transactions.csv")
    print(f"  {'Atributo':<25} {'Dtype':<12} {'Tipo':<12} {'#Únicos':>8} {'%Nulos':>8}")
    print("  " + "-" * 70)

    linhas = []
    for col in df_t.columns:
        tipo = classificar_atributo(df_t[col])
        n_unicos = df_t[col].nunique()
        pct_nulo = df_t[col].isna().mean() * 100
        marcador = "  (*ID)" if col in COLUNAS_ID else ""
        print(f"  {col:<25} {str(df_t[col].dtype):<12} {tipo:<12} "
              f"{n_unicos:>8,} {pct_nulo:>7.2f}%{marcador}")
        linhas.append({"Atributo": col, "Tipo": tipo})

    df_tax = pd.DataFrame(linhas)
    contagem = df_tax["Tipo"].value_counts()
    print("\n  --> RESUMO:")
    for tipo, qtd in contagem.items():
        print(f"    {tipo:<12}: {qtd} atributo(s)")

    plots.plot_tipos_atributos(df_t, contagem)

    print("\n\n\n")

    return df_tax

def variavel_alvo(df_t: pd.DataFrame):
    separador("ANÁLISE DA VARIÁVEL ALVO - \"is_fraud\"")

    col_alvo = "is_fraud"

    vc = df_t[col_alvo].value_counts()
    vc_pct = df_t[col_alvo].value_counts(normalize=True) * 100
 
    print(f"\n  {'Classe':<10} {'Contagem':>12} {'%':>8}")
    print("  " + "-" * 34)
    for cls in vc.index:
        print(f"  {str(cls):<10} {vc[cls]:>12,} {vc_pct[cls]:>7.2f}%")
 
    razao = vc.max() / vc.min()
    print(f"\n  Razão de desbalanceamento: {razao:.1f}:1")
    if razao > 3:
        print(f"  --> A variável alvo ( {col_alvo} ) está desbalanceada")
    elif razao > 1.5:
        print(f"  --> A variável alvo ( {col_alvo} ) apresenta algum desbalanceamento.")
    else:
        print(f"  A variável alvo ( {col_alvo} ) parece razoavelmente balanceada.")

    plots.plot_variavel_alvo(df_t, vc, vc_pct, col_alvo)

    print("\n\n\n")

    return col_alvo

def tipos_fraude(df_t: pd.DataFrame, col_alvo: str):
    separador("ANÁLISE DOS TIPOS DE FRAUDE")

    col_tipo = next((c for c in df_t.columns if "type" in c.lower() or "kind" in c.lower()), None)
                     
    if col_tipo and col_tipo != col_alvo:
        print(f"  Tipos de fraude ('{col_tipo}'):")
        
        vc = df_t[col_tipo].value_counts()
        vc_pct = df_t[col_tipo].value_counts(normalize=True) * 100
        
        for cat in vc.index:
            print(f"    {str(cat):<20}: {vc[cat]:>10,} ({vc_pct[cat]:>5.2f}%)")

    plots.plot_tipos_fraude(df_t, col_tipo, col_alvo)

    print("\n\n\n")

    return col_tipo

def distribuicoes_numericas(df_t: pd.DataFrame, col_alvo: str):
    separador("DISTRIBUIÇÕES DOS ATRIBUTOS NUMÉRICOS")

    num_cols = df_t.select_dtypes(include=np.number).columns.tolist()
    if col_alvo in num_cols and df_t[col_alvo].nunique() <= 10:
        num_cols = [c for c in num_cols if c != col_alvo]

    if not num_cols:
        print("  Nenhum atributo numérico encontrado.")
        return

    n = len(num_cols)
    ncols_plot = min(3, n)
    nrows_plot = (n + ncols_plot - 1) // ncols_plot

    print("\n  Assimetria (skewness) dos atributos numéricos:")
    print(f"    {'Atributo':<25} {'Skewness':>10}  {'Interpretação'}")
    print("    " + "-" * 55)
    for col in num_cols:
        sk = df_t[col].skew()
        if abs(sk) < 0.5:
            interp = "simétrico"
        elif abs(sk) < 1:
            interp = "assimétrico moderado"
        else:
            interp = "assimétrico forte"
        print(f"    {col:<25} {sk:>10.4f}  {interp}")
    print("\n")

    plots.plot_distribuicoes_numericas(df_t, col_alvo, nrows_plot, ncols_plot, num_cols)
    plots.plot_boxplots_por_classe(df_t, col_alvo, nrows_plot, ncols_plot, num_cols)

    print("\n\n\n")

def qualidade_dados(df_t: pd.DataFrame, df_p: pd.DataFrame, df_f: pd.DataFrame):
    separador("QUALIDADE DOS DADOS")

    print("\n  --> VALORES AUSENTES")
    tem_nulos = False
    for nome, df in [("transactions.csv", df_t), ("account_profiles.csv", df_p), ("fraud_patterns.csv", df_f)]:
        df_nulos = (pd.DataFrame({"Ausentes": df.isna().sum(), "%": df.isna().mean() * 100}).query("Ausentes > 0").sort_values("%", ascending=False))
        if df_nulos.empty:
            print(f"    -> {nome}: Sem valores ausentes")
        else:
            tem_nulos = True
            print(f"    -> {nome}: {len(df_nulos)} coluna(s) com ausentes")
            
            print(f"    {'Atributo':<25} {'Ausentes':>12}  {'%':>8}")
            print("    " + "-" * 47)
            
            for col_name, row in df_nulos.iterrows():
                ausentes = int(row['Ausentes'])
                pct = row['%']
                print(f"    {col_name:<25} {ausentes:>12,}  {pct:>7.2f}%")

        print()

    if tem_nulos:
        plots.plot_valores_ausentes(df_t, df_p, df_f)

    print("\n\n  --> DUPLICATAS")
    for nome, df in [("transactions.csv", df_t), ("account_profiles.csv", df_p), ("fraud_patterns.csv", df_f)]:
        dup = df.duplicated().sum()
        if dup > 0:
            print(f"    -> {nome}: {dup:,} linhas duplicadas ({dup/len(df)*100:.3f}%)")
        else:
            print(f"    -> {nome}:  Nenhuma duplicata.")

    print("\n\n  --> DETECÇÃO DE OUTLIERS (Regra 3-sigma)")
    num_cols = df_t.select_dtypes(include=np.number).columns
    dados_outliers = []

    for col in num_cols:
        serie = df_t[col].dropna()
        mu = serie.mean()
        sigma = serie.std()
        outliers = ((serie < mu - 3 * sigma) | (serie > mu + 3 * sigma)).sum()
        pct = (outliers / len(serie)) * 100
        
        if outliers > 0: 
            dados_outliers.append({ "Atributo": col, "Outliers": outliers, "%": pct })

    if dados_outliers:
        df_out = pd.DataFrame(dados_outliers).sort_values("%", ascending=False)
        print(f"  Encontrados outliers em {len(df_out)} colunas:")
        print("  " + "-" * 50)
        print(df_out.to_string(index=False))
        
        plots.plot_outliers_por_atributo(df_t, df_out["Atributo"].tolist(), df_out)
    else:
        print("    Nenhum outlier severo detectado pela regra 3-sigma.")

    col_data = next((c for c in df_t.columns if any(p in c.lower() for p in ["date", "time", "timestamp", "data", "hora"])), None)
    if col_data:
        print(f"\n\n  --> VERIFICAÇÃO TEMPORAL ('{col_data}')")
        try:
            datas = pd.to_datetime(df_t[col_data], errors="coerce")
            print(f"  Data mínima : {datas.min()}")
            print(f"  Data máxima : {datas.max()}")
            if datas.isna().sum() > 0:
                print(f"  Datas nulas : {datas.isna().sum():,}")

            plots.plot_volume_temporal(df_t, col_data, datas)
        except Exception as e:
            print(f"  -> Não foi possível processar coluna de data: {e}")

    print("\n\n\n")

def correlacoes(df_t: pd.DataFrame, col_alvo: str):
    separador("CORRELAÇÕES ENTRE ATRIBUTOS")

    num_cols = df_t.select_dtypes(include=np.number).columns.tolist()

    corr = df_t[num_cols].corr()

    plots.plot_correlacao_heatmap(num_cols, corr)

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

    if col_alvo and col_alvo in df_t.columns:
        try:
            corr_alvo = df_t[num_cols].corrwith(pd.to_numeric(df_t[col_alvo], errors="coerce")).sort_values(key=abs, ascending=False)

            print(f"\n  Correlação dos atributos numéricos com '{col_alvo}':")
            print(f"  {'Atributo':<30} {'Correlação':>12}")
            print("  " + "-" * 44)
            for attr, val in corr_alvo.items():
                if attr != col_alvo:
                    print(f"  {attr:<30} {val:>12.4f}")

            plots.plot_correlacao_com_alvo(corr_alvo, col_alvo)
        except Exception as e:
            print(f"  -> Não foi possível calcular correlação com alvo: {e}")

    print("\n\n\n")

def analise_monetaria(df_t: pd.DataFrame, col_alvo: str, col_tipo: str):
    separador("ANÁLISE DO VALOR DAS TRANSAÇÕES")

    col_valor = "amount"
    if col_valor in df_t.columns:
        desc = df_t[col_valor].describe(percentiles=[0.5, 0.99])
        
        # Resumo Geral
        print(f"  -> Visão Geral ('{col_valor}'):")
        print(f"    Média   : {desc['mean']:>12,.2f}")
        print(f"    Mediana : {desc['50%']:>12,.2f}")
        print(f"    P. 99%  : {desc['99%']:>12,.2f}")
        print(f"    Máximo  : {desc['max']:>12,.2f}")


        if col_alvo in df_t.columns:
            print(f"\n  Estatísticas por Classe ('{col_alvo}'):")
            print(f"    {'Classe':<14} {'Contagem':>12} {'Média':>14} {'Mediana':>14} {'Máximo':>14}")
            print("    " + "-" * 72)
            
            desc_grp = df_t.groupby(col_alvo)[col_valor].describe(percentiles=[0.5])
            for cls, row in desc_grp.iterrows():
                nome_cls = "0 (Legítima)" if str(cls) == "0" else "1 (Fraude)" if str(cls) == "1" else str(cls)
                
                contagem = int(row['count'])
                media = row['mean']
                mediana = row['50%']
                maximo = row['max']
                print(f"    {nome_cls:<14} {contagem:>12,} {media:>14,.2f} {mediana:>14,.2f} {maximo:>14,.2f}")

            print()

            plots.plot_valor_por_classe(df_t, col_alvo, col_valor)

        if col_tipo and col_tipo in df_t.columns and col_tipo != col_alvo:
            plots.plot_mediana_valor_por_tipo(df_t, col_tipo, col_valor)
    
    print("\n\n\n")

def sintese_preprocessamento(df_t: pd.DataFrame, col_alvo: str):
    separador("SÍNTESE DA ANÁLISE E PLANO DE AÇÃO")

    n_linhas, n_cols = df_t.shape
    num_cols = df_t.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df_t.select_dtypes(include=["object", "category"]).columns if "id" not in c.lower()]
    
    
    nulos = df_t.isna().sum()
    cols_com_nulos = nulos[nulos > 0].index.tolist()
    dups = df_t.duplicated().sum()
    
    
    vc = df_t[col_alvo].value_counts()
    razao = vc.max() / vc.min()
    pct_fraude = (vc.min() / n_linhas) * 100

    if col_alvo in num_cols:
        num_cols_no_alvo = [c for c in num_cols if c != col_alvo]
        corr = df_t[num_cols_no_alvo].corrwith(df_t[col_alvo]).abs().sort_values(ascending=False)
        top_preditores = ", ".join(corr.head(3).index.tolist())
    else:
        top_preditores = "N/A"

    print("    -> RESUMO GERAL DO DATASET:")
    print(f"      Dimensões         : {n_linhas:,} registros x {n_cols} colunas")
    print(f"      Tipagem           : {len(num_cols)} numéricas | {len(cat_cols)} categóricas")
    print(f"      Saúde dos Dados   : {dups} duplicatas | {len(cols_com_nulos)} coluna(s) com nulos")
    print(f"      Variável Alvo     : {pct_fraude:.2f}% de fraudes (Desbalanceamento severo de {razao:.1f}:1)")
    print(f"      Top Preditores    : {top_preditores}")

    print("\n\n\n")

def executar_eda(df_transactions: pd.DataFrame, df_account_profiles: pd.DataFrame, df_fraud_patterns: pd.DataFrame):

    caracterizacao(df_transactions, df_account_profiles, df_fraud_patterns)
    col_alvo = variavel_alvo(df_transactions)
    col_tipo = tipos_fraude(df_transactions, col_alvo)
    distribuicoes_numericas(df_transactions, col_alvo)
    qualidade_dados(df_transactions, df_account_profiles, df_fraud_patterns)
    correlacoes(df_transactions, col_alvo)
    analise_monetaria(df_transactions, col_alvo, col_tipo)
    sintese_preprocessamento(df_transactions, col_alvo)

    print("Análise Exploratória de Dados concluída! Imagens salvas em:", plots.OUTPUT_DIR)
