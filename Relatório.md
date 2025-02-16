# **Instituto Federal do Ceará - Campus Maracanaú**  
## Inteligência Artificial(IA)
**Professor:** Amaurí Holanda
### **Dupla:**
**Aluno:** Francisco Aldenor Silva Neto  
**Aluno:** Isabelly Pinheiro da Costa

# Questão 1: Regressão Linear. 
Implemente um modelo de regressão linear. Para isso, utilize um conjunto de dados sintético gerado com a equação:
<p align="center">
y = 3x + 5 + ε(1)
</p>
onde x segue distribuição uniforme entre -10 e 10 e ε ́e um ruído gaussiano com média zero e desvio padrão de 2.

Faça os seguintes passos:

1. Gere um conjunto de dados com pelo menos 100 pontos.
```python
N = 100
x = np.random.uniform(-10, 10, N)  # Distribuição uniforme entre -10 e 10
epsilon = np.random.normal(0, 2, N)  # Ruído gaussiano com média 0 e desvio 2
y = 3 * x + 5 + epsilon  # Equação dada

# Convertendo para arrays 2D para regressão
X = np.vstack((np.ones(N), x)).T  # Adicionando termo de viés
```
2. Divida os dados em treino (80%) e teste (20%).
```python
train_size = int(0.8 * N)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
```
3. Implemente modelos de regressão linear empregando:
  - a solução de mínimos quadrados (pseudo-inversa);
  ```python
  theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

  # Previsões no conjunto de teste
  y_pred_pinv = X_test @ theta
  ```
  - uma rede neural com uma camada treinada via gradiente descendente utilizando MSE-Loss (Erro Quadrático Médio) e otimizador SGD.
  ```python
  # Implementação com Rede Neural
  class LinearRegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # Apenas uma camada linear

    def forward(self, x):
        return self.linear(x)

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

  # Predições da Rede Neural
  y_pred_nn = model(X_test_torch).detach().numpy()  
  ```
4. Apresente as soluções para cada um dos métodos acima.
```python
print("Coeficientes da regressão MSE:", theta)
print("Coeficientes da Rede Neural: w =", w_nn, ", b =", b_nn)
```
Coeficientes da regressão MSE: [4.8363477  2.93680725]
Coeficientes da Rede Neural: w = 2.9368069171905518 , b = 4.8363356590271

5. Avalie o desempenho dos modelos e visualize os resultados.
```python
# Cálculo dos Erros Médios Quadráticos (MSE)
mse_pinv = mean_squared_error(y_test, y_pred_pinv)
mse_nn = mean_squared_error(y_test, y_pred_nn)

print(f"MSE (Pseudo-Inversa): {mse_pinv:.4f}")
print(f"MSE (Rede Neural): {mse_nn:.4f}")

plt.scatter(x, y, label="Dados reais", alpha=0.6)
plt.plot(x[train_size:], y_pred_pinv, label="Pseudo-Inversa", color="red")
plt.plot(x[train_size:], y_pred_nn, label="Rede Neural", color="green", linestyle="dashed")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Regressão Linear: Pseudo-Inversa vs Rede Neural")
plt.savefig("imagens/Q1-regressao_linear.png")
plt.show()
```
MSE (Pseudo-Inversa): 3.9022
MSE (Rede Neural): 3.9022
![Q1-regressao_linear](imagens/Q1-regressao_linear.png)  
O gráfico mostra a comparação entre os modelos de regressão linear obtidos com a solução de mínimos quadrados e a rede neural.

---

# Questão 2 - Regressão Logística para Classificação Binária. 
Implemente um modelo de regressão logística para resolver um problema de classificação binária utilizando um conjunto de dados sintético.

Faça os seguintes passos:
1. Utilize a função make classification da biblioteca Scikit-Learn para gerar um conjunto de dados com 500 amostras, 2 variáveis preditoras e 2 classes.
2. Divida os dados em treino (70%) e teste (30%).
3. Implemente um modelo de regressão logística (i.e., rede neural com uma  ́unica camada de saída e ativação sigmoid).
4. Treine o modelo utilizando gradiente descendente (versão não-estocástica) (conforme visto em sala).
5. Avalie a acurácia no conjunto de teste e visualize a fronteira de decisão do classificador.


**Resultados:**

- **Acurácia no Conjunto de Teste:**  
  Acurácia: 0.8733

**Gráfico Gerado:**  
![Q2-fronteira_decisao_logistica](imagens/Q2-fronteira_decisao_logistica.png)  
O gráfico apresenta a fronteira de decisão do classificador de regressão logística.

---

## Questão 3: Rede Neural MLP para Classificação Binária

**Objetivo:**  
Implementar uma rede neural do tipo MLP para a tarefa de classificação binária, utilizando o conjunto de dados `make_moons`.

**Resultados:**

- **Melhor Número de Neurônios:**  
  Melhor número de neurônios: 50 (Menor Loss: 0.0325)

- **Acurácia no Conjunto de Teste:**  
  Acurácia: 0.9867

**Gráficos Gerados:**  
![q3-evolucao_perda](imagens/q3-evolucao_perda.png)  
O gráfico mostra a evolução da função de perda (loss) ao longo do treinamento.

![q3-fronteira_decisao_mlp](imagens/q3-fronteira_decisao_mlp.png)  
O gráfico apresenta a fronteira de decisão do classificador MLP com o melhor número de neurônios na camada oculta.

---

## Questão 4: Rede Neural para Classificação de Imagens do Conjunto MNIST

**Objetivo:**  
Implementar uma rede neural para a classificação de imagens do conjunto MNIST.

**Resultados:**

- **Acurácia no Conjunto de Teste:**  
  Acurácia: 0.9625

**Gráficos Gerados:**  
![Q4_Predicoes](imagens/Q4_Predicoes.png)  
O gráfico exibe algumas previsões feitas pelo modelo, mostrando imagens e suas respectivas classes previstas.

![Q4-Acuracia_treinamento](imagens/Q4-Acuracia_treinamento.png)  
O gráfico mostra a evolução da acurácia durante o treinamento.

---

## Conclusão

Neste trabalho, foram implementados e avaliados diferentes modelos para tarefas de regressão e classificação. A regressão linear foi aplicada com duas abordagens, utilizando a solução de mínimos quadrados e uma rede neural simples. O modelo de regressão logística foi implementado para um problema de classificação binária, e a rede neural MLP foi aplicada para a classificação binária no conjunto `make_moons`. Finalmente, a rede neural MLP foi treinada no conjunto MNIST para a tarefa de classificação de imagens.

A análise dos gráficos gerados durante o treinamento e a avaliação final demonstraram o bom desempenho dos modelos, com destaque para a acurácia do modelo de rede neural no conjunto de testes, especialmente na questão 3 (MLP) e na questão 4 (MNIST).
