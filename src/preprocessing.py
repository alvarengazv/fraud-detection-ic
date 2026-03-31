import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def executar_preprocessing(df_transactions: pd.DataFrame, df_account_profiles: pd.DataFrame, df_fraud_patterns: pd.DataFrame):
    # Criar dataset que será o principal para análise
    df = df_transactions.copy()
    
    # Remover colunas de ID (transaction_id, account_id)
    df = df.drop(columns=["transaction_id", "account_id"])
    
    # Remover colunas que podem causar target leakage
    df = df.drop(columns=["fraud_pattern"])
    
    # Remover colunas redundantes
    df = df.drop(columns=["timestamp", "is_weekend", "amount", "mcc_code"])

    # One-hot encoding em variáveis categóricas (dtype=int para compatibilidade com XGBoost)
    df = pd.get_dummies(df, columns=["merchant_category", "merchant_country", "device_type"], dtype=int)

    # Exportar dataset pré-processado para CSV
    csv_path = "./dataset/preprocessed.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Dataset pré-processado salvo em: {csv_path}")
    print(f"  Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    
    return df