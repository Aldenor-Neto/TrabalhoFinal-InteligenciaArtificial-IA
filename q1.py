import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

# Gerando os dados sintéticos
np.random.seed(42)
torch.manual_seed(42)

N = 100
x = np.random.uniform(-10, 10, N)  # Distribuição uniforme entre -10 e 10
epsilon = np.random.normal(0, 2, N)  # Ruído gaussiano com média 0 e desvio 2
y = 3 * x + 5 + epsilon  # Equação dada

# Convertendo para arrays 2D para regressão
X = np.vstack((np.ones(N), x)).T  # Adicionando termo de viés

# Dividindo em treino (80%) e teste (20%)
train_size = int(0.8 * N)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 2. Regressão Linear via Pseudo-Inversa (Mínimos Quadrados)
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print("Coeficientes da regressão (Mínimos Quadrados):", theta)

# Previsões no conjunto de teste
y_pred_pinv = X_test @ theta

# Implementação com Rede Neural
class LinearRegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Apenas uma camada linear

    def forward(self, x):
        return self.linear(x)

# Convertendo os dados para Tensores do PyTorch 
X_train_torch = torch.tensor(X_train[:, 1], dtype=torch.float32).view(-1, 1)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_torch = torch.tensor(X_test[:, 1], dtype=torch.float32).view(-1, 1)

# Criando o modelo, função de perda e otimizador
model = LinearRegressionNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Treinamento da Rede Neural
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train_torch)
    loss = criterion(y_pred, y_train_torch)
    loss.backward()
    optimizer.step()

# Obtendo os coeficientes treinados
w_nn, b_nn = model.linear.weight.item(), model.linear.bias.item()
print("Coeficientes da Rede Neural: w =", w_nn, ", b =", b_nn)

# Predições da Rede Neural
y_pred_nn = model(X_test_torch).detach().numpy()

# Cálculo dos Erros Médios Quadráticos (MSE)
mse_pinv = mean_squared_error(y_test, y_pred_pinv)
mse_nn = mean_squared_error(y_test, y_pred_nn)

print(f"MSE (Pseudo-Inversa): {mse_pinv:.4f}")
print(f"MSE (Rede Neural): {mse_nn:.4f}")

os.makedirs("imagens", exist_ok=True)

plt.scatter(x, y, label="Dados reais", alpha=0.6)
plt.plot(x[train_size:], y_pred_pinv, label="Pseudo-Inversa", color="red")
plt.plot(x[train_size:], y_pred_nn, label="Rede Neural", color="green", linestyle="dashed")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Regressão Linear: Pseudo-Inversa vs Rede Neural")
plt.savefig("imagens/Q1-regressao_linear.png")
plt.show()
