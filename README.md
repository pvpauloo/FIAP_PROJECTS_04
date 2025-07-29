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
import yfinance as yf

symbol = 'ITUB'
start_date = '2018-01-01'
end_date = '2024-07-01'

df = yf.download(symbol, start=start_date, end=end_date)
```

---

#### ⚙️ Normalização e separação dos dados

Os dados foram normalizados com `MinMaxScaler` e divididos em conjuntos de treino e teste com base em uma janela de 60 dias:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Close']])

sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # formato LSTM
```

---

#### 🧠 Criação do modelo LSTM

O modelo foi criado com `Keras` utilizando 2 camadas LSTM e uma camada densa de saída:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)
```

---

#### 📊 Avaliação com métricas MAE, RMSE e MAPE

Após treinar o modelo, realizamos a previsão no conjunto de teste e avaliamos as métricas de erro:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

# y_test e y_pred são valores reais vs previstos
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")
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
