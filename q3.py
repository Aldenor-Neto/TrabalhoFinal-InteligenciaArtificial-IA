import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

os.makedirs("imagens", exist_ok=True)

# Gerando os dados sintéticos
np.random.seed(42)
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

# Dividindo os dados (70% treino, 15% validação, 15% teste)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convertendo para tensores PyTorch
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Criando DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)


# MLP
class MLP(nn.Module):
    def __init__(self, n_hidden):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, n_hidden)  
        self.output = nn.Linear(n_hidden, 1)  

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Ativação ReLU
        x = torch.sigmoid(self.output(x))  # Ativação Sigmoid
        return x


# Função para treinar o modelo
def train_model(n_hidden, epochs=100, lr=0.01):
    model = MLP(n_hidden)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Avaliação no conjunto de validação
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(X_v), y_v).item() for X_v, y_v in val_loader) / len(val_loader)
        val_losses.append(val_loss)

    return model, train_losses, val_losses


# Testando diferentes números de neurônios na camada oculta
neurons_list = [5, 10, 20, 50]
best_model = None
best_val_loss = float('inf')
best_n = None

plt.figure(figsize=(10, 5))
for n in neurons_list:
    model, train_losses, val_losses = train_model(n_hidden=n)

    # Salvando o melhor modelo
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_model = model
        best_n = n

    plt.plot(val_losses, label=f"{n} neurônios")

plt.xlabel("Épocas")
plt.ylabel("Loss (Erro)")
plt.legend()
plt.title("Evolução da Função Custo (Validação)")
plt.savefig("imagens/q3-evolucao_perda.png")
plt.show()

print(f"Melhor número de neurônios: {best_n} (Menor Loss: {best_val_loss:.4f})")

# Avaliação no conjunto de teste
best_model.eval()
with torch.no_grad():
    y_pred = (best_model(X_test) >= 0.5).float()
accuracy = (y_pred.eq(y_test).sum() / len(y_test)).item()
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
X_grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

with torch.no_grad():
    Z = best_model(X_grid).numpy().reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test.numpy().flatten(), cmap="coolwarm", edgecolors="k")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Fronteira de Decisão - Melhor Modelo ({best_n} Neurônios)")
plt.savefig("imagens/q3-fronteira_decisao_mlp.png")
plt.show()
