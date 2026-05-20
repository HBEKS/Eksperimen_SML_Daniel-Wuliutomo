# automate_Daniel.py
# Script untuk preprocessing otomatis dataset Loan Approval

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

def load_data(filepath):
    """
    Load dataset from CSV file
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape}")
    return df

def preprocess_data(df):
    """
    Preprocess loan approval dataset
    """
    print("\n" + "="*50)
    print("STARTING PREPROCESSING")
    print("="*50)
    
    # --- 1. Menghapus Data Duplikat ---
    print("\n" + "="*50)
    print("PROSES REMOVE DUPLICATE ROWS")
    print("="*50)
    print(f"Shape awal dataset: {df.shape}")
    print(f"Jumlah baris duplikat yang ditemukan: {df.duplicated().sum()}")
    
    df_clean = df.drop_duplicates(keep='first')
    
    print(f"Shape dataset SETELAH data duplikat dihapus: {df_clean.shape}")
    print(f"Cek ulang duplikat: {df_clean.duplicated().sum()}")
    
    # --- 2. Menangani Data Kosong (Missing Values) ---
    print(f"\n2. Missing values sebelum handling: {df_clean.isnull().sum().sum()}")
    
    # Cek missing values per kolom
    missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
    if missing_cols:
        print(f"Kolom dengan missing values: {missing_cols}")
    
    # Isi missing values numerik dengan median
    for col in df_clean.select_dtypes(include=['float64', 'int64']).columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            print(f"   - {col}: diisi dengan median")
    
    # Isi missing values kategorikal dengan modus
    for col in df_clean.select_dtypes(include=['object']).columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].mode().empty:
                df_clean[col] = df_clean[col].fillna(0)
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            print(f"   - {col}: diisi dengan modus")
    
    print(f"   Missing values setelah handling: {df_clean.isnull().sum().sum()}")
    
    # --- 3. Encoding Data Kategorikal ---
    print("\n3. Encoding data kategorikal:")
    
    le = LabelEncoder()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"   Kolom kategorikal: {categorical_cols}")
        for col in categorical_cols:
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            print(f"   - {col}: telah di-encode")
    else:
        print("   Tidak ada kolom kategorikal")
    
    # --- 4. Normalisasi/Standarisasi Fitur ---
    print("\n4. Normalisasi fitur:")
    
    # Pisahkan fitur dan target
    target_col = 'loan_status'
    X = df_clean.drop(target_col, axis=1)
    y = df_clean[target_col]
    
    # Scale fitur numerik dengan StandardScaler
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if num_cols:
        X[num_cols] = scaler.fit_transform(X[num_cols])
        print(f"   - {len(num_cols)} fitur numerik di-scale dengan StandardScaler")
        print(f"     Fitur: {', '.join(num_cols)}")
    else:
        print("   - Tidak ada fitur numerik yang perlu di-scale")
    
    # Gabungkan kembali
    df_processed = X.copy()
    df_processed[target_col] = y
    
    return df_processed, scaler

def save_results(df_processed, scaler, output_dir):
    """
    Save preprocessed data and scaler
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan CSV
    output_path = os.path.join(output_dir, 'loan_processed.csv')
    df_processed.to_csv(output_path, index=False)
    print(f"\n📁 CSV saved: {output_path}")
    
    return output_path

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Preprocess Loan Approval Dataset')
    parser.add_argument('--input', type=str, default='dataset/loan_data.csv',
                        help='Input dataset path (default: dataset/loan_data.csv)')
    parser.add_argument('--output', type=str, default='./preprocessing/loan_preprocessing',
                        help='Output directory (default: ./preprocessing/loan_preprocessing)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input)
    
    # Preprocess data
    df_processed, scaler = preprocess_data(df)
    
    # Save results
    output_path = save_results(df_processed, scaler, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ PREPROCESSING SELESAI!")
    print("="*60)
    print(f"📊 Shape akhir dataset: {df_processed.shape}")
    print(f"\n📈 Target distribution:")
    print(f"   - Approved (1): {(df_processed['loan_status']==1).sum():,}")
    print(f"   - Not Approved (0): {(df_processed['loan_status']==0).sum():,}")
    print(f"\n💾 Output saved to: {args.output}")

if __name__ == "__main__":
    main()