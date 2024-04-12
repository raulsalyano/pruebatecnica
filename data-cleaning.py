import pandas as pd

# Cargar el archivo CSV en un DataFrame
file_path = '/content/credit_risk_dataset.csv'
df = pd.read_csv(file_path)

# Mostrar información general del DataFrame
print(df.info())

# Identificar y manejar valores nulos/faltantes
print(df.isnull().sum())

# Verificar y manejar valores nulos en las columnas numéricas
numeric_cols = df.select_dtypes(include='number').columns.tolist()  # Obtener columnas numéricas

# Crear un objeto SimpleImputer para reemplazar valores nulos con la media
imputer = SimpleImputer(strategy='mean')

# Iterar sobre las columnas numéricas y reemplazar los valores nulos con la media
for col in numeric_cols:
    if df[col].isnull().any():  # Verificar si la columna tiene valores nulos
        df[col] = imputer.fit_transform(df[[col]])

# Verificar si todavía hay valores nulos después del reemplazo
print(df.isnull().sum())


df.to_csv('/content/drive/MyDrive/ruta/a/tu/archivo_limpiado.csv', index=False)
