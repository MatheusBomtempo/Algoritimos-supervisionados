import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# Configurar matplotlib para usar fonte que suporte caracteres acentuados
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Traduzir nomes das classes
nomes_classes = ['Setosa', 'Versicolor', 'Virginica']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar e treinar o modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# Calcular métricas de desempenho
acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Realizar validação cruzada
cv_scores = cross_val_score(rf_model, X, y, cv=5)

# Imprimir resultados
print("Métricas de desempenho:")
print(f"Acurácia: {acuracia:.4f}")
print(f"Precisão: {precisao:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Acurácia média (validação cruzada): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Criar matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotar matriz de confusão
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
plt.colorbar()
tick_marks = np.arange(len(nomes_classes))
plt.xticks(tick_marks, nomes_classes, rotation=45)
plt.yticks(tick_marks, nomes_classes)
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")

# Adicionar valores à matriz de confusão
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('matriz_confusao.png')
plt.close()

# Plotar importância das características
feature_importance = rf_model.feature_importances_
feature_names = ['Comprimento Sépala', 'Largura Sépala', 'Comprimento Pétala', 'Largura Pétala']

plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.xticks(range(len(feature_importance)), feature_names, rotation=45)
plt.title("Importância das Características")
plt.xlabel("Características")
plt.ylabel("Importância")
plt.tight_layout()
plt.savefig('importancia_caracteristicas.png')
plt.close()

print("Os gráficos foram salvos como 'matriz_confusao.png' e 'importancia_caracteristicas.png'.")