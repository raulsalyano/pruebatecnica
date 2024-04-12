# Importar librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el archivo CSV en un DataFrame
file_path = '/content/archivo_limpiado.csv'
df = pd.read_csv(file_path)

# Mostrar información general del DataFrame
print(df.info())

#Seleccionar las columnas categóricas para codificar
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Aplicar codificación one-hot a las columnas categóricas
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Convertir las entradas de las columnas binarias a 1 y 0
df_encoded = df_encoded.astype(int)

# Mostrar las primeras filas del DataFrame codificado
print(df_encoded.head())


# Seleccionar solo las columnas numéricas para la normalización
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Aplicar la normalización Min-Max a las columnas numéricas
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Mostrar las primeras filas del DataFrame normalizado
print(df.head())

# Seleccionar solo las columnas numéricas para calcular estadísticas descriptivas
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Calcular estadísticas descriptivas utilizando el método describe()
stats_descriptive = df[numeric_cols].describe()

# Mostrar las estadísticas descriptivas
print(stats_descriptive)

##   Procesamiento de datos

# Variables categóricas de interés
categorical_vars = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Contar los valores únicos en cada variable categórica
for var in categorical_vars:
    print(f"Variable: {var}")
    print(df[var].value_counts())
    print()  # Salto de línea para separar las salidas

# Configuración opcional para mejorar la visualización
plt.figure(figsize=(10, 8))
sns.set(style="whitegrid")

# Visualizar la distribución de las variables categóricas mediante gráficos de barras
for i, var in enumerate(categorical_vars, 1):
    plt.subplot(2, 2, i)
    sns.countplot(data=df, x=var)
    plt.title(f"Distribución de {var}")

####Correlación  entre la edad y los ingresos
# Seleccionar las variables de interés
x = df['person_age']
y = df['person_income']

# Calcular el coeficiente de correlación de Pearson
correlation = np.corrcoef(x, y)[0, 1]

# Crear un gráfico de dispersión
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5, color='blue')

# Agregar detalles al gráfico
plt.title(f"Correlación entre person_age y person_income\nCoeficiente de correlación: {correlation:.2f}")
plt.xlabel('Edad (person_age)')
plt.ylabel('Ingreso (person_income)')
plt.grid(True)

# Mostrar la línea de tendencia (opcional)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--")

# Mostrar el gráfico
plt.show()

### Prueba t de Student para el estado del préstamo y los años de employment
# Filtrar y limpiar los datos relevantes
df_filtered = df[['loan_status', 'person_emp_length']].dropna()

# Dividir los datos en dos grupos según loan_status (0 y 1)
group_0 = df_filtered[df_filtered['loan_status'] == 0]['person_emp_length']
group_1 = df_filtered[df_filtered['loan_status'] == 1]['person_emp_length']

# Realizar la prueba t de Student independiente
t_statistic, p_value = stats.ttest_ind(group_0, group_1)

# Mostrar los resultados
print("Resultados de la prueba t de Student:")
print(f"Valor t: {t_statistic}")
print(f"Valor p: {p_value}")

# Interpretar el resultado
alpha = 0.05  # Nivel de significancia
if p_value < alpha:
    print("Se rechaza la hipótesis nula. Hay una diferencia significativa en los años de empleo entre los préstamos completados y no completados.")
else:
    print("No se puede rechazar la hipótesis nula. No hay suficiente evidencia para afirmar una diferencia significativa en los años de empleo entre los préstamos completados y no completados.")

###  Predicción de incumplimiento de préstamos

# Seleccionar características y objetivo
features = df_encoded.drop(['loan_status'], axis=1)  # Todas las columnas excepto 'loan_status'
target = df_encoded['loan_status']  # Variable objetivo: 'loan_status'

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Escalar características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Modelo de regresión lineal

# Crear y entrenar el modelo de regresión logística
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_scaled, y_train)

# Evaluar el modelo en el conjunto de prueba
accuracy_logistic = logistic_model.score(X_test_scaled, y_test)
print(f"Puntuación de precisión del modelo de regresión logística: {accuracy_logistic}")

# Predicciones del modelo de regresión logística
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Matriz de confusión y reporte de clasificación
print("Matriz de confusión del modelo de regresión logística:")
print(confusion_matrix(y_test, y_pred_logistic))
print("\nReporte de clasificación del modelo de regresión logística:")
print(classification_report(y_test, y_pred_logistic))
