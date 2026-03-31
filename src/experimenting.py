import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, precision_recall_curve, auc,
                             roc_curve, confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

warnings.filterwarnings("ignore")

OUTPUT_DIR = "./images/experimenting_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO AUXILIAR
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_recall(prec_array, rec_array, target_recall=0.80):
    """Retorna a maior precisão onde recall >= target_recall."""
    mask = rec_array >= target_recall
    if mask.any():
        return prec_array[mask].max()
    return 0.0


def _criar_modelos():
    """Retorna um dicionário com todos os modelos a avaliar."""
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42),
        "Reg. Logística (sem norm.)": LogisticRegression(random_state=42, max_iter=1000),
        "Reg. Logística (com norm.)": Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# VALIDAÇÃO CRUZADA (K-FOLD)
# ══════════════════════════════════════════════════════════════════════════════

def executar_kfold(X_train_full, y_train_full, n_splits=5):
    """Executa validação cruzada estratificada e retorna DataFrame com resultados."""

    print("\n" + "="*70)
    print("  VALIDAÇÃO CRUZADA (Stratified K-Fold)")
    print("="*70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    modelos = _criar_modelos()

    # Dicionário: nome -> lista de PR-AUC por fold
    resultados = {nome: [] for nome in modelos}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"  Fold {fold + 1}/{n_splits} ...", end=" ")

        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]

        for nome, modelo in modelos.items():
            modelo.fit(X_train_fold, y_train_fold)
            proba = modelo.predict_proba(X_val_fold)[:, 1]
            p, r, _ = precision_recall_curve(y_val_fold, proba)
            resultados[nome].append(auc(r, p))

        print("concluído")

    # ── Montar DataFrame de resultados ────────────────────────────────────────
    linhas = []
    for nome, scores in resultados.items():
        for fold_i, score in enumerate(scores):
            linhas.append({"Modelo": nome, "Fold": fold_i + 1, "PR-AUC": score})
    df_kfold = pd.DataFrame(linhas)

    # ── Resumo no terminal ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTADOS DA VALIDAÇÃO CRUZADA (PR-AUC)")
    print("="*70 + "\n")

    resumo = df_kfold.groupby("Modelo")["PR-AUC"].agg(["mean", "std"])
    for nome in modelos:
        row = resumo.loc[nome]
        print(f"  {nome:<30} Média: {row['mean']:.4f} | Desvio Padrão: {row['std']:.4f}")

    # ── Salvar CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "kfold_resultados.csv")
    df_kfold.to_csv(csv_path, index=False)
    print(f"\n  [salvo] {csv_path}")

    # ── Gráfico PR-AUC por fold ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    cores = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71"]

    for i, nome in enumerate(modelos):
        scores = resultados[nome]
        ax.plot(range(1, n_splits + 1), scores, '-o', label=nome,
                color=cores[i % len(cores)], linewidth=2, markersize=8)
        ax.axhline(np.mean(scores), linestyle='--', color=cores[i % len(cores)],
                   alpha=0.4, linewidth=1)

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("PR-AUC", fontsize=12)
    ax.set_title("PR-AUC por Fold — Validação Cruzada", fontsize=14, fontweight="bold")
    ax.set_xticks(range(1, n_splits + 1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    img_path = os.path.join(OUTPUT_DIR, "kfold_pr_auc.png")
    fig.savefig(img_path, bbox_inches="tight")
    print(f"  [salvo] {img_path}")
    plt.show()

    # ── Boxplot comparativo ───────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    dados_box = [resultados[nome] for nome in modelos]
    nomes_box = list(modelos.keys())

    bp = ax2.boxplot(dados_box, patch_artist=True, notch=True,
                     medianprops={"color": "black", "linewidth": 2})
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.6)

    ax2.set_xticklabels(nomes_box, fontsize=9, rotation=15, ha="right")
    ax2.set_ylabel("PR-AUC", fontsize=12)
    ax2.set_title("Boxplot — PR-AUC por Modelo (K-Fold)", fontsize=14, fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()

    img_path2 = os.path.join(OUTPUT_DIR, "kfold_boxplot.png")
    fig2.savefig(img_path2, bbox_inches="tight")
    print(f"  [salvo] {img_path2}")
    plt.show()

    return df_kfold


# ══════════════════════════════════════════════════════════════════════════════
# TREINAMENTO FINAL + AVALIAÇÃO NO TESTE
# ══════════════════════════════════════════════════════════════════════════════

def executar_experimentacao(df: pd.DataFrame):
    print("\n" + "="*70)
    print("  SEÇÃO 4 — EXPERIMENTAÇÃO")
    print("="*70)

    # 1. Separar dados de Teste (separados e intocáveis)
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # 2. Perguntar se deseja rodar K-Fold
    resposta = input("\n  Deseja executar a validação cruzada (K-Fold)? [s/N]: ").strip().lower()
    if resposta == "s":
        executar_kfold(X_train_full, y_train_full)

    # 3. Treinamento Final
    print("\n" + "="*70)
    print("  TREINAMENTO FINAL")
    print("="*70)

    modelos = _criar_modelos()

    for nome, modelo in modelos.items():
        print(f"  Treinando {nome} ...", end=" ")
        modelo.fit(X_train_full, y_train_full)
        print("concluído")

    # 4. Previsões e métricas
    resultados = {}

    for nome, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1]

        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        resultados[nome] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "prec_curve": prec_curve,
            "rec_curve": rec_curve,
            "fpr": fpr,
            "tpr": tpr,
            "Acurácia": accuracy_score(y_test, y_pred),
            "Precisão": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "PR-AUC": auc(rec_curve, prec_curve),
            "KS Statistic": max(tpr - fpr),
            "P@80%Recall": precision_at_recall(prec_curve, rec_curve),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
            "Matriz de Confusão": confusion_matrix(y_test, y_pred),
        }

    # 5. Salvar resultados finais em CSV
    metricas_csv = ["Acurácia", "Precisão", "Recall", "F1-Score",
                    "PR-AUC", "KS Statistic", "P@80%Recall", "ROC-AUC"]

    linhas_csv = []
    for nome, res in resultados.items():
        linha = {"Modelo": nome}
        for m in metricas_csv:
            linha[m] = round(res[m], 4)
        linhas_csv.append(linha)

    df_resultados = pd.DataFrame(linhas_csv)
    csv_path = os.path.join(OUTPUT_DIR, "resultados_teste_final.csv")
    df_resultados.to_csv(csv_path, index=False)
    print(f"\n  [salvo] {csv_path}")

    # 6. Apresentação de resultados
    for i, (nome, res) in enumerate(resultados.items(), 1):
        print("\n" + "="*70)
        print(f"  MODELO {i} - {nome.upper()}")
        print("="*70)
        for m in metricas_csv:
            print(f"  {m:<20}: {res[m]:.4f}")
        print(f"  Matriz de Confusão:\n  {res['Matriz de Confusão']}")

    # 6. Curvas ROC e PR
    print("\n" + "="*70)
    print("  CURVAS ROC E PR")
    print("="*70)

    cores = ["#2C3E50", "#E74C3C", "#3498DB", "#2ECC71"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for i, (nome, res) in enumerate(resultados.items()):
        cor = cores[i % len(cores)]
        ax1.plot(res["fpr"], res["tpr"],
                 label=f'{nome} (AUC={res["ROC-AUC"]:.4f})', color=cor)
        ax2.plot(res["rec_curve"], res["prec_curve"],
                 label=f'{nome} (PR-AUC={res["PR-AUC"]:.4f})', color=cor)

    ax1.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.set_title('Curva ROC')
    ax1.legend(loc='lower right', fontsize=8)

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precisão')
    ax2.set_title('Curva PR')
    ax2.legend(loc='lower left', fontsize=8)

    fig.tight_layout()
    caminho = os.path.join(OUTPUT_DIR, "curvas_roc_pr.png")
    fig.savefig(caminho, bbox_inches="tight")
    print(f"\n  [salvo] {caminho}")
    plt.show()

    # 7. Matrizes de Confusão
    print("\n" + "="*70)
    print("  MATRIZES DE CONFUSÃO")
    print("="*70)

    labels = ["Legítima", "Fraude"]
    n_modelos = len(resultados)
    fig, axes = plt.subplots(1, n_modelos, figsize=(6 * n_modelos, 5))
    if n_modelos == 1:
        axes = [axes]

    for ax, (nome, res) in zip(axes, resultados.items()):
        cm = res["Matriz de Confusão"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".4f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax, vmin=0, vmax=1,
                    annot_kws={"size": 14, "fontweight": "bold"})
        ax.set_title(f"{nome}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

    fig.suptitle("Matrizes de Confusão Normalizadas", fontsize=15, fontweight="bold")
    fig.tight_layout()
    caminho = os.path.join(OUTPUT_DIR, "matrizes_confusao.png")
    fig.savefig(caminho, bbox_inches="tight")
    print(f"\n  [salvo] {caminho}")
    plt.show()
