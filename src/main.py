
# Importações
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import shutil
import pandas as pd

# Imports locais
from eda.eda import executar_eda
import preprocessing
import experimenting

def load_data():
    path = kagglehub.dataset_download(
        "sergionefedov/fraud-detection-1m-transactions-7-fraud-types"
    )

    destino = "./dataset"
    os.makedirs(destino, exist_ok=True)

    for item in os.listdir(path):
        origem_item = os.path.join(path, item)
        destino_item = os.path.join(destino, item)

        if os.path.isdir(origem_item):
            shutil.copytree(origem_item, destino_item, dirs_exist_ok=True)
        else:
            shutil.copy2(origem_item, destino_item)

    print(f"Arquivos copiados para {destino}")

    df_transactions = pd.read_csv(f"{destino}/transactions.csv")
    df_account_profiles = pd.read_csv(f"{destino}/account_profiles.csv")
    df_fraud_patterns = pd.read_csv(f"{destino}/fraud_patterns.csv")

    print("Datasets carregados com sucesso!")
    print(f"\nTransações: ./dataset/transactions.csv")
    print(f"Perfis de contas: ./dataset/account_profiles.csv")
    print(f"Padrões de fraude: ./dataset/fraud_patterns.csv")
    
    return df_transactions, df_account_profiles, df_fraud_patterns

def clear_console():
    print("\n"*50)
    if os.name == 'nt': 
        os.system('cls')
    else: 
       os.system('clear')

def main():
    clear_console()

    print("="*100)
    print("ATIVIDADE PRÁTICA 1 - METODOLOGIA EXPERIMENTAL - INTELIGÊNCIA COMPUTACIONAL")
    print("="*100 + "\n")

    print("--> Dataset escolhido foi: Fraud Detection - 1M Transactions (7 Fraud Types)")
    print()

    preprocessed_path = "./dataset/preprocessed.csv"

    # ── Verificar se já existe dataset pré-processado ─────────────────────
    if os.path.exists(preprocessed_path):
        print(f"--> Dataset pré-processado encontrado: {preprocessed_path}")
        resp = input("  Deseja usar o dataset pré-processado existente? [S/n]: ").strip().lower()
        if resp != "n":
            df = pd.read_csv(preprocessed_path)
            print(f"  Carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas\n")
        else:
            df = None
    else:
        df = None

    # ── Carregar dados brutos se necessário ───────────────────────────────
    if df is None:
        print("\n--> Carregando dataset bruto...")
        df_transactions, df_account_profiles, df_fraud_patterns = load_data()

        # EDA
        resp_eda = input("\n--> Deseja executar a Análise Exploratória (EDA)? [s/N]: ").strip().lower()
        if resp_eda == "s":
            clear_console() 
            print("\n" + "-"*40)
            print("  ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
            print("-"*40)
            executar_eda(df_transactions, df_account_profiles, df_fraud_patterns)
            input("\n--> Pressione Enter para continuar para o pré-processamento...")
            clear_console()

        # Pré-processamento
        resp_prep = input("\n--> Deseja executar o Pré-processamento? [S/n]: ").strip().lower()
        if resp_prep == "n":
            print("  --> Sem pré-processamento, não é possível continuar a experimentação.")
            return
        else:
            clear_console() 
            print("\n" + "-"*40)
            print("  PRÉ-PROCESSAMENTO")
            print("-"*40)
            df = preprocessing.executar_preprocessing(df_transactions, df_account_profiles, df_fraud_patterns)

    input("\n--> Pressione Enter para continuar para a metodologia experimental...")
    clear_console() 

    # Metodologia Experimental
    print("\n" + "-"*40)
    print("  METODOLOGIA EXPERIMENTAL")
    print("-"*40)
    experimenting.executar_experimentacao(df)
    
    print("\n" + "-"*50)

if __name__ == "__main__":
    main()
    