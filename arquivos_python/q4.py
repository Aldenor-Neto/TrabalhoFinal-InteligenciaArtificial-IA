import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Carregar o dataset MNIST e normalizar
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Dividir o conjunto de treino em treino (80%) e validação (20%)
train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# Rede neural MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten (28x28 -> 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Sem softmax, pois CrossEntropyLoss já inclui softmax
        return x


# Inicializar modelo, função de perda e otimizador
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinar o modelo e armazenar as métricas
num_epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, correct_train, total_train = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images, labels

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_losses.append(running_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)

    # Validação
    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images, labels
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_losses.append(val_loss / len(valid_loader))
    val_accuracies.append(correct_val / total_val)

    print(
        f"Época [{epoch + 1}/{num_epochs}] - Loss Treino: {train_losses[-1]:.4f}, Acurácia Treino: {train_accuracies[-1]:.4f} - Loss Validação: {val_losses[-1]:.4f}, Acurácia Validação: {val_accuracies[-1]:.4f}")

# Avaliação no conjunto de teste
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images, labels
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f"Acurácia no conjunto de teste: {test_accuracy:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Acurácia Treino", marker='o')
plt.plot(range(1, num_epochs + 1), val_accuracies, label="Acurácia Validação", marker='s')
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.title("Evolução da Acurácia durante o Treinamento")
plt.legend()
plt.grid()
plt.savefig("imagens/Q4-Acuracia_treinamento.png", dpi=300)
plt.show()

model.eval()
dataiter = iter(test_loader)
images, labels = next(dataiter)  

# Imagens e previsões
outputs = model(images)
_, predicted = torch.max(outputs, 1)

fig = plt.figure(figsize=(10, 5))
for idx in range(6):
    ax = fig.add_subplot(2, 3, idx + 1)
    ax.imshow(images[idx].numpy().squeeze(), cmap='gray')
    ax.set_title(f'Predição: {predicted[idx].item()}')
    ax.axis('off')
    plt.savefig("imagens/Q4_Predicoes.png", dpi=300)  
plt.show()

