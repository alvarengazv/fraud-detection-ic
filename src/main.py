
# Importações
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import shutil
import pandas as pd

# Imports locais
from eda.eda import executar_eda
# import preprocessing
# import modeling

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

def space_return(fase):
    print("\n" + "-"*50 + "\n")
    if fase:
        print(f"Deseja iniciar a fase de {fase}? (s/n)")
        inicio = input().strip().lower()
        if inicio != 's':
            print(f"Fase de {fase} pulada. Pressione Enter para continuar...")  
            return False

        print(f"---- Pressione Enter para continuar... ----")
        clear_console()
        return True
        
    else:
        print(f"---- Pressione Enter para continuar... ----")
    input()
    print("-"*50 + "\n")

def main():
    clear_console()

    print("="*100)
    print("ATIVIDADE PRÁTICA 1 - METODOLOGIA EXPERIMENTAL - INTELIGÊNCIA COMPUTACIONAL")
    print("="*100 + "\n")

    print("--> Dataset escolhido foi: Fraud Detection - 1M Transactions (7 Fraud Types)")
    print("\n")
    
    print("--> Carregando dataset...")
    df_transactions, df_account_profiles, df_fraud_patterns = load_data()

    if space_return("Análise Exploratória de Dados"):
        print("\n" + "-"*40)
        print("  ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
        print("-"*40)
        executar_eda(df_transactions, df_account_profiles, df_fraud_patterns)

    # # Pré-processamento
    # print("\nIniciando Pré-processamento...")
    # preprocessing.executar_preprocessing(df)

    # print("\n" + "-"*50)

    # # Modelagem
    # print("\nIniciando Modelagem...")
    # modeling.executar_modelagem(df)

if __name__ == "__main__":
    main()