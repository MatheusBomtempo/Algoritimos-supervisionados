import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Criar e treinar o modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = knn_model.predict(X_test)

# Calcular métricas de desempenho
acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Realizar validação cruzada
cv_scores = cross_val_score(knn_model, X, y, cv=5)

# Imprimir resultados
print("Métricas de desempenho do KNN:")
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
plt.title("Matriz de Confusão - KNN")
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
plt.savefig('matriz_confusao_knn.png')
plt.close()

# Plotar importância das características (usando uma abordagem alternativa para KNN)
feature_names = ['Comprimento Sépala', 'Largura Sépala', 'Comprimento Pétala', 'Largura Pétala']

# Calcular a importância das características baseada na acurácia de cada característica individualmente
importancia_caracteristicas = []
for i in range(X.shape[1]):
    knn_single = KNeighborsClassifier(n_neighbors=5)
    score = cross_val_score(knn_single, X[:, i:i+1], y, cv=5).mean()
    importancia_caracteristicas.append(score)

# Normalizar as importâncias
importancia_caracteristicas = np.array(importancia_caracteristicas)
importancia_caracteristicas = (importancia_caracteristicas - importancia_caracteristicas.min()) / (importancia_caracteristicas.max() - importancia_caracteristicas.min())

plt.figure(figsize=(12, 6))
plt.bar(range(len(importancia_caracteristicas)), importancia_caracteristicas)
plt.xticks(range(len(importancia_caracteristicas)), feature_names, rotation=45)
plt.title("Importância das Características - KNN")
plt.xlabel("Características")
plt.ylabel("Importância Relativa")
plt.tight_layout()
plt.savefig('importancia_caracteristicas_knn.png')
plt.close()

print("Os gráficos foram salvos como 'matriz_confusao_knn.png' e 'importancia_caracteristicas_knn.png'.")

# Comparação com diferentes valores de K
k_values = range(1, 21)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

plt.figure(figsize=(12, 6))
plt.plot(k_values, cv_scores, 'bo-')
plt.title("Desempenho do KNN com Diferentes Valores de K")
plt.xlabel("Número de Vizinhos (K)")
plt.ylabel("Acurácia Média (Validação Cruzada)")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.savefig('desempenho_k_valores.png')
plt.close()

print("O gráfico de desempenho para diferentes valores de K foi salvo como 'desempenho_k_valores.png'.")