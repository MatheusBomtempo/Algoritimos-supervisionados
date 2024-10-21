import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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

# Padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar e treinar o modelo SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred = svm_model.predict(X_test_scaled)

# Calcular métricas de desempenho
acuracia = accuracy_score(y_test, y_pred)
precisao = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Realizar validação cruzada
cv_scores = cross_val_score(svm_model, scaler.fit_transform(X), y, cv=5)

# Imprimir resultados
print("Métricas de desempenho do SVM:")
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
plt.title("Matriz de Confusão - SVM")
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
plt.savefig('matriz_confusao_svm.png')
plt.close()

# Plotar importância das características (usando os coeficientes do SVM)
feature_names = ['Comprimento Sépala', 'Largura Sépala', 'Comprimento Pétala', 'Largura Pétala']

# Para SVM com kernel RBF, usaremos a média dos vetores de suporte como uma medida de importância
support_vectors = svm_model.support_vectors_
importancia_caracteristicas = np.abs(np.mean(support_vectors, axis=0))
importancia_caracteristicas /= np.sum(importancia_caracteristicas)

plt.figure(figsize=(12, 6))
plt.bar(range(len(importancia_caracteristicas)), importancia_caracteristicas)
plt.xticks(range(len(importancia_caracteristicas)), feature_names, rotation=45)
plt.title("Importância das Características - SVM")
plt.xlabel("Características")
plt.ylabel("Importância Relativa")
plt.tight_layout()
plt.savefig('importancia_caracteristicas_svm.png')
plt.close()

print("Os gráficos foram salvos como 'matriz_confusao_svm.png' e 'importancia_caracteristicas_svm.png'.")

# Comparação com diferentes valores de C (parâmetro de regularização)
C_values = [0.1, 1, 10, 100]
cv_scores_C = []

for C in C_values:
    svm = SVC(kernel='rbf', C=C, random_state=42)
    scores = cross_val_score(svm, scaler.fit_transform(X), y, cv=5, scoring='accuracy')
    cv_scores_C.append(scores.mean())

plt.figure(figsize=(12, 6))
plt.semilogx(C_values, cv_scores_C, 'bo-')
plt.title("Desempenho do SVM com Diferentes Valores de C")
plt.xlabel("Parâmetro de Regularização (C)")
plt.ylabel("Acurácia Média (Validação Cruzada)")
plt.xticks(C_values, C_values)
plt.grid(True)
plt.tight_layout()
plt.savefig('desempenho_C_valores.png')
plt.close()

print("O gráfico de desempenho para diferentes valores de C foi salvo como 'desempenho_C_valores.png'.")