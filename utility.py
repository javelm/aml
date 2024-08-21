import pandas as pd

def remove_outliers_iqr(df):
    df_clean = df.copy()
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):  # Check if column is numeric
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        else:
            print(f"Skipping non-numeric column: {column}")
    return df_clean