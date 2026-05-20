import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, RobustScaler

def run_preprocessing(input_path, output_dir):
    print(f"Memulai preprocessing untuk data: {input_path}")
    
    # 1. Memuat Dataset
    df = pd.read_csv(input_path)
    print(f"Shape awal: {df.shape}")
    
    # 2. Menghapus Duplikat
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"Duplikat dihapus: {initial_shape[0] - df.shape[0]} rows")
    
    # 3. Menangani Missing Values (dataset ini bersih, tapi tetap jalanin)
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # 4. Normalisasi Fitur (Amount pakai RobustScaler karena outlier)
    print("Melakukan normalisasi fitur...")
    
    # Scale Amount dengan RobustScaler
    amount_scaler = RobustScaler()
    df['Amount'] = amount_scaler.fit_transform(df[['Amount']])
    
    # Scale fitur lainnya dengan StandardScaler
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = [col for col in num_cols if col != 'Class']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # 5. Menyimpan Hasil
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'creditcard_preprocessed.csv')
    df.to_csv(output_file, index=False)
    
    print(f"Preprocessing selesai. Shape akhir: {df.shape}")
    print(f"Data disimpan di: {output_file}")
    return output_file

if __name__ == "__main__":
    INPUT_FILE = "dataset/creditcard.csv"
    OUTPUT_FOLDER = "preprocessing/creditcard_preprocessing"
    
    run_preprocessing(INPUT_FILE, OUTPUT_FOLDER)