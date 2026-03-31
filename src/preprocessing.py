import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# hour_of_day,day_of_week,amount,merchant_category,mcc_code,merchant_country,card_present,device_type,device_known,ip_risk_score,is_foreign_txn,time_since_last_s,velocity_1h,amount_vs_avg_ratio,account_age_days,has_2fa,credit_limit,is_fraud,fraud_pattern


def executar_preprocessing(df_transactions: pd.DataFrame, df_account_profiles: pd.DataFrame, df_fraud_patterns: pd.DataFrame):
    print("\n" + "="*70)
    print("  SEÇÃO 3 — PRÉ-PROCESSAMENTO DE DADOS")
    print("="*70)

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

    # Printar quantas e quais colunas foram criadas
    # print(f"\nColunas criadas: {df.shape[1]}")
    # print(f"Colunas: {df.columns.tolist()}")

    # 43 colunas

    # Debug
    # print(df.head())
    # Exportar dataset pré-processado para CSV
    csv_path = "./dataset/preprocessed.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Dataset pré-processado salvo em: {csv_path}")
    print(f"  Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    
    return df