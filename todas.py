import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Configurar matplotlib para usar fonte que suporte caracteres acentuados
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Traduzir nomes das classes e características
nomes_classes = ['Setosa', 'Versicolor', 'Virginica']
nomes_caracteristicas = ['Comprimento Sépala', 'Largura Sépala', 'Comprimento Pétala', 'Largura Pétala']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializar os modelos
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
svm_model = SVC(kernel='rbf', random_state=42)

# Treinar os modelos
rf_model.fit(X_train, y_train)
knn_model.fit(X_train_scaled, y_train)
svm_model.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred_rf = rf_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_svm = svm_model.predict(X_test_scaled)


# Função para calcular métricas
def calcular_metricas(y_true, y_pred):
    return {
        'Acurácia': accuracy_score(y_true, y_pred),
        'Precisão': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-score': f1_score(y_true, y_pred, average='weighted')
    }


# Calcular métricas para cada modelo
metricas_rf = calcular_metricas(y_test, y_pred_rf)
metricas_knn = calcular_metricas(y_test, y_pred_knn)
metricas_svm = calcular_metricas(y_test, y_pred_svm)

# Criar DataFrame com as métricas
df_metricas = pd.DataFrame({
    'Random Forest': metricas_rf,
    'KNN': metricas_knn,
    'SVM': metricas_svm
})

print("Comparação de Métricas:")
print(df_metricas)

# Plotar gráfico de barras comparativo
plt.figure(figsize=(12, 6))
df_metricas.plot(kind='bar')
plt.title("Comparação de Métricas entre Random Forest, KNN e SVM")
plt.xlabel("Métricas")
plt.ylabel("Valor")
plt.legend(title="Algoritmos")
plt.tight_layout()
plt.savefig('comparacao_metricas.png')
plt.close()


# Função para plotar matriz de confusão
def plot_confusion_matrix(cm, titulo, nome_arquivo):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(titulo)
    plt.colorbar()
    tick_marks = np.arange(len(nomes_classes))
    plt.xticks(tick_marks, nomes_classes, rotation=45)
    plt.yticks(tick_marks, nomes_classes)
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.close()


# Plotar matrizes de confusão
plot_confusion_matrix(confusion_matrix(y_test, y_pred_rf), "Matriz de Confusão - Random Forest",
                      'matriz_confusao_rf.png')
plot_confusion_matrix(confusion_matrix(y_test, y_pred_knn), "Matriz de Confusão - KNN", 'matriz_confusao_knn.png')
plot_confusion_matrix(confusion_matrix(y_test, y_pred_svm), "Matriz de Confusão - SVM", 'matriz_confusao_svm.png')

# Importância das características para Random Forest
importancia_rf = rf_model.feature_importances_

# Importância das características para KNN (usando acurácia individual)
importancia_knn = []
for i in range(X.shape[1]):
    knn_single = KNeighborsClassifier(n_neighbors=5)
    score = cross_val_score(knn_single, X[:, i:i + 1], y, cv=5).mean()
    importancia_knn.append(score)
importancia_knn = np.array(importancia_knn)
importancia_knn = (importancia_knn - importancia_knn.min()) / (importancia_knn.max() - importancia_knn.min())

# Importância das características para SVM (usando magnitude dos coeficientes para kernel linear)
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
importancia_svm = np.abs(svm_linear.coef_).sum(axis=0)
importancia_svm /= importancia_svm.sum()

# Plotar importância das características
plt.figure(figsize=(12, 6))
width = 0.25
x = np.arange(len(nomes_caracteristicas))
plt.bar(x - width, importancia_rf, width, label='Random Forest', alpha=0.7)
plt.bar(x, importancia_knn, width, label='KNN', alpha=0.7)
plt.bar(x + width, importancia_svm, width, label='SVM', alpha=0.7)
plt.xlabel('Características')
plt.ylabel('Importância')
plt.title('Comparação da Importância das Características')
plt.xticks(x, nomes_caracteristicas, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('comparacao_importancia_caracteristicas.png')
plt.close()

print("Todos os gráficos comparativos foram salvos.")