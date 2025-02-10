import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

os.makedirs("imagens", exist_ok=True)

#  Gerando os dados sintéticos
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=2, n_classes=2, n_redundant=0, random_state=42)

#  Dividindo os dados em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#  Implementando a Regressão Logística com Gradiente Descendente
class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Adicionando um termo de viés (coluna de 1s)
        X = np.c_[np.ones(X.shape[0]), X]
        self.theta = np.zeros(X.shape[1])

        # Treinamento via Gradiente Descendente Batch
        for _ in range(self.epochs):
            z = X @ self.theta
            h = self.sigmoid(z)
            gradient = (1 / len(y)) * X.T @ (h - y)
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]  
        return (self.sigmoid(X @ self.theta) >= 0.5).astype(int)

# Criando e treinando o modelo
model = LogisticRegressionGD(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# Avaliando a acurácia no conjunto de teste
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

# Fronteira de decisão
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", edgecolors="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Fronteira de decisão - Acurácia: {accuracy:.4f}")
plt.savefig("imagens/Q2-fronteira_decisao_logistica.png")
plt.show()

