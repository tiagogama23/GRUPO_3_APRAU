import pandas as pd
import numpy as np
import os


def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Aplica o pré-processamento padrão ao dataset.

    Passos:
    1. Corrigir focus_factor (formato vírgula -> ponto)
    2. Consolidar duration_1..5 em coluna única 'duration'
    3. Remover features problemáticas
    """
    df = pd.read_csv(input_path)

    # Corrigir focus_factor
    if df['focus_factor'].dtype == 'object':
        df['focus_factor'] = df['focus_factor'].str.replace(',', '.').astype(float)

    # Consolidar duration_1..5
    duration_cols = ['duration_1', 'duration_2', 'duration_3', 'duration_4', 'duration_5']
    if all(c in df.columns for c in duration_cols):
        df['duration'] = df[duration_cols].values @ np.arange(1, 6)
        df = df.drop(columns=duration_cols)

    # Remover features problemáticas
    cols_to_drop = [
        'echo_constant',
        'is_dance_hit',
        'temp_zscore',
        'signal_power',
        'duration_log',
        'target_regression'
    ]
    cols_found = [c for c in cols_to_drop if c in df.columns]
    if cols_found:
        df = df.drop(columns=cols_found)

    # Guardar CSV
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "group_3.csv")
    output_file = os.path.join(script_dir, "group_3_preprocessed.csv")

    df = preprocess_dataset(input_file, output_file)

    print(f"Dataset processado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"Guardado em: {output_file}")
