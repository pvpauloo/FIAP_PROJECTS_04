# 📈 Previsão de Fechamento de Ações com LSTM e FastAPI

Este projeto é o resultado do **Tech Challenge da Fase 4 da Pós-Tech MLET FIAP**, cujo desafio consistia em desenvolver uma pipeline completa de **Deep Learning com LSTM** para prever o valor de fechamento de ações de uma empresa à escolha, fazer o **deploy do modelo** e disponibilizá-lo via **API** em produção.

## 💼 Desafio proposto (resumo)
De acordo com o documento oficial [Pos_Tech_Fase4.pdf], os requisitos incluíam:
- Coletar e pré-processar dados históricos de ações
- Construir e treinar um modelo preditivo com LSTM
- Salvar o modelo treinado para inferência
- Fazer o deploy com **SageMaker**
- Disponibilizar uma API RESTful (FastAPI ou Flask)
- Monitorar a performance do modelo com **Amazon CloudWatch**

---

## 🔁 Pipeline de Desenvolvimento

### 1. 📊 Coleta, Pré-processamento e Treinamento (`criacao_modelo.ipynb`)

#### 📥 Coleta de dados com `yfinance`

Utilizamos a biblioteca `yfinance` para coletar dados da ação **ITUB4.SA**, considerando um intervalo de aproximadamente 5 anos:

```python
ACAO = 'ITUB'            # Símbolo da empresa (ex: 'DIS' para Disney)
START_DATE = '2021-01-01'  # Data de início da coleta de dados
END_DATE = '2025-07-01'    # Data de fim da coleta de dados
SEQUENCE = 60       # Número de dias passados para prever o próximo dia (timestep do LSTM)

df = yf.download(ACAO, start=START_DATE, end=END_DATE)
```

---

#### ⚙️ Normalização e separação dos dados

Os dados foram normalizados com `MinMaxScaler` e divididos em conjuntos de treino e teste com base em uma janela de 60 dias:

```python
def create_sequences(dataset, seq_len):
    X, y = [], []
    for i in range(len(dataset) - seq_len):
        X.append(dataset[i:(i + seq_len), 0])
        y.append(dataset[i + seq_len, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQUENCE)

train_size = int(len(X) * 0.8) # 80/20
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape dos dados para o formato esperado pelo LSTM: [samples, timesteps, features]
# Onde features é 1 porque estamos usando apenas a coluna 'Close'
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
```

---

#### 🧠 Criação do modelo LSTM

O modelo foi criado com `Keras` utilizando 2 camadas LSTM e uma camada densa de saída:

```python
model = Sequential()
# Primeira camada LSTM: retorna sequências para a próxima camada LSTM
model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE, 1)))
model.add(Dropout(0.2)) # 20% dos neurônios são desativados aleatoriamente para regularização

# Segunda camada LSTM: não retorna sequências, pois é a última camada LSTM antes da camada de saída
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2)) # Mais dropout para regularização

# Camada de saída densa: um único neurônio para prever o preço de fechamento
model.add(Dense(units=1))

# Compilar o modelo:
model.compile(optimizer='adam', loss='mean_squared_error')

# Exibir um resumo da arquitetura do modelo
model.summary()
```

---

#### 🧠 Treinamento do modelo LSTM

```python
history = model.fit(
    X_train, y_train,
    epochs=50,           # Número de épocas (ajuste conforme necessário)
    batch_size=32,       # Tamanho do lote (ajuste conforme necessário)
    validation_split=0.1, # 10% dos dados de treino serão usados para validação durante o treinamento
    verbose=1            # Exibir progresso do treinamento
)
```
#### 📊 Avaliação com métricas MAE, RMSE e MAPE

Após treinar o modelo, realizamos a previsão no conjunto de teste e avaliamos as métricas de erro:

```python
# Inverter a normalização para obter os valores reais dos preços
predictions = scaler.inverse_transform(predictions_scaled)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcular métricas de avaliação
mae = mean_absolute_error(y_test_unscaled, predictions)
rmse = math.sqrt(mean_squared_error(y_test_unscaled, predictions))

# Calcular MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test_unscaled - predictions) / y_test_unscaled)) * 100

print(f"Métricas de Avaliação no Conjunto de Teste:")
print(f"  MAE (Mean Absolute Error): {mae:.2f}")
print(f"  RMSE (Root Mean Square Error): {rmse:.2f}")
print(f"  MAPE (Mean Absolute Percentage Error): {mape}%")
```

Essas métricas nos ajudaram a entender a precisão e o comportamento do modelo.

---

### 2. 🚀 Deploy no SageMaker (`deploy_modelo.ipynb`)
- Salvamos o modelo em formato `SavedModel` do TensorFlow
- Criamos o endpoint no **Amazon SageMaker**
- Validamos o endpoint com `boto3` usando uma entrada de teste

---

### 3. 📈 Monitoramento com CloudWatch (`dashboard.png`)
Criamos um dashboard no **Amazon CloudWatch** com as seguintes métricas:
- Utilização de Disco, CPU e Memória
- Quantidade de Invocações
- Latência do Modelo
- Taxa de Erros
- Requisições Concorrentes

![Dashboard CloudWatch](https://github.com/pvpauloo/FIAP_PROJECTS_04/blob/main/dashboard.png?raw=true)

---

### 4. 🔌 API com FastAPI (`API_main.py`)
Desenvolvemos uma API hospedada em instância EC2 com as seguintes rotas:

#### `GET /`
Página inicial com resumo da API

#### `GET /fechamento/hoje`
Retorna a previsão do fechamento de **hoje (D+0)** com base nos últimos 60 dias úteis de fechamento.

#### `GET /fechamento/proximos`
Retorna a previsão para os **próximos 5 dias úteis (D+1 até D+5)**, fazendo previsão iterativa.

**Exemplo de chamada do modelo com boto3:**

```python
response = runtime.invoke_endpoint(
    EndpointName=ENDPOINT,
    ContentType="application/json",
    Body=json.dumps(payload),
)
result = json.loads(response["Body"].read())
```

---

## 🔀 Desenho Arquitetura AWS

![Arquitetura](https://github.com/pvpauloo/FIAP_PROJECTS_04/blob/main/aws_arch_img.png?raw=true)

---
## 📁 Estrutura de Arquivos

| Arquivo | Descrição |
|--------|-----------|
| `criacao_modelo.ipynb` | ETL, construção e avaliação do modelo |
| `deploy_modelo.ipynb` | Deploy e testes com endpoint no SageMaker |
| `dashboard.png` | Captura do painel de monitoramento no CloudWatch |
| `API_main.py` | Código da API FastAPI com chamadas ao modelo |
| `Pos_Tech_Fase4.pdf` | Enunciado do desafio proposto |

---

## ✅ Conclusão

O projeto atendeu todos os requisitos propostos:
- Modelo LSTM funcional com bom desempenho
- Deploy efetivo no SageMaker
- API responsiva com inferência real
- Monitoramento ativo em produção

Este projeto demonstra habilidades práticas em **Machine Learning aplicado a séries temporais**, **deploy na nuvem**, e **engenharia de APIs** com observabilidade.

---

> Projeto desenvolvido como parte da Pós-Tech MLET – Fase 4 (FIAP)
