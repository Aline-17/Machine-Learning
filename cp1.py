# ==============================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ==============================
# CARREGAMENTO DO DATASET
# ==============================
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Primeiras linhas do dataset:")
print(df.head())

# ==============================
# 🔹 PARTE 1 - ANÁLISE EXPLORATÓRIA
# ==============================

print("\n==============================")
print("VERIFICAÇÃO DE VALORES NaN")
print("==============================")
print(df.isnull().sum())

print("\n==============================")
print("ESTATÍSTICAS DESCRITIVAS")
print("==============================")
print(df.describe())

print("\n==============================")
print("MEDIANA")
print("==============================")
print(df.median())

print("\n==============================")
print("VARIÂNCIA")
print("==============================")
print(df.var())

print("\n==============================")
print("DESVIO PADRÃO")
print("==============================")
print(df.std())

print("\n==============================")
print("VALORES MÍNIMOS")
print("==============================")
print(df.min())

print("\n==============================")
print("VALORES MÁXIMOS")
print("==============================")
print(df.max())

# ==============================
# BOXPLOT PARA OUTLIERS
# ==============================
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("Detecção de Outliers (Boxplot)")
plt.show()

# ==============================
# 🔹 PARTE 2 - MACHINE LEARNING (KNN)
# ==============================

# Separação de variáveis
X = df.drop('target', axis=1)
y = df['target']

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Teste de diferentes valores de K
k_values = [1, 3, 5, 7, 9]
results = []

print("\n==============================")
print("TREINAMENTO DO MODELO KNN")
print("==============================")

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results.append((k, acc))
    print(f"K = {k} -> Acurácia = {acc:.2f}")

# ==============================
# TABELA DE RESULTADOS
# ==============================
results_df = pd.DataFrame(results, columns=['K', 'Acurácia'])

print("\n==============================")
print("TABELA FINAL DE RESULTADOS")
print("==============================")
print(results_df)

# ==============================
# GRÁFICO DE DESEMPENHO
# ==============================
plt.figure(figsize=(8,5))
plt.plot(results_df['K'], results_df['Acurácia'], marker='o')
plt.title("Acurácia vs Valor de K")
plt.xlabel("Valor de K")
plt.ylabel("Acurácia")
plt.grid()
plt.show()