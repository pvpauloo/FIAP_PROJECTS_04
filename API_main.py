import boto3
import yfinance as yf
import numpy as np
import json
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from typing import List

# --- Configuração geral ---
ACAO = 'ITUB4.SA'
SEQUENCE = 60
DIAS_EXTRA = 90
REGIAO = 'us-east-2'
ENDPOINT = 'tensorflow-inference-2025-07-26-19-17-42-056'

# --- Inicia FastAPI ---
app = FastAPI(
    title="API de Previsão de Fechamento ITUB4",
    description="Previsões de fechamento com modelo LSTM hospedado no SageMaker",
    version="1.0.0"
)

# --- Funções auxiliares ---
def get_ultimos_60_fechamentos() -> List[float]:
    hoje = datetime.now().date()
    inicio = hoje - timedelta(days=DIAS_EXTRA)
    df = yf.download(ACAO, start=inicio, end=hoje)

    if df.empty or 'Close' not in df:
        raise ValueError("Erro ao obter dados da ação.")

    fechamentos = df['Close'].dropna().values.tolist()

    if len(fechamentos) < SEQUENCE + 1:
        raise ValueError(f"Dados insuficientes ({len(fechamentos)}) para gerar sequência de {SEQUENCE}.")

    return fechamentos[-(SEQUENCE+1):-1]

def prever_um_dia(sequencia_entrada: List[float], runtime) -> float:
    payload = {"instances": [sequencia_entrada]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    return result[0] if isinstance(result, list) else result

def prever_proximos_dias(n_dias=5) -> list[float]:
    sequencia = get_ultimos_60_fechamentos()
    runtime = boto3.client("sagemaker-runtime", region_name=REGIAO)
    previsoes = []

    for _ in range(n_dias):
        # chama a função central que executa o modelo corretamente
        pred = prever_um_dia(sequencia, runtime)
        previsoes.append(pred)
        print("Pred:  ")
        print([pred.get('predicted_price')])
        # atualiza a sequência para a próxima chamada
        print("sequencias:  ")
        sequencia = sequencia[1:] + [[pred.get('predicted_price')]]
        print(sequencia)

    return previsoes

# --- Rota principal (home) ---
@app.get("/", response_class=HTMLResponse, summary="Página inicial da API")
def home():
    html_content = """
    <html>
        <head>
            <title>API de Previsão ITUB4</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f6f8; color: #333; }
                h1 { color: #005288; }
                code { background-color: #eaeaea; padding: 2px 4px; border-radius: 4px; }
                .rotas { margin-top: 20px; }
                .rota { margin-bottom: 16px; padding: 12px; background: #fff; border-left: 5px solid #005288; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body>
            <h1>📈 API de Previsão de Fechamento - ITUB4</h1>
            <p>Esta API utiliza um modelo LSTM treinado para prever o valor de fechamento da ação <strong>ITUB4.SA</strong> (Itaú Unibanco),
            hospedado no <strong>Amazon SageMaker</strong>.</p>

            <h2>🧠 Modelo</h2>
            <ul>
                <li>Entrada: Últimos <code>60</code> dias úteis de fechamento</li>
                <li>Saída: Previsão de preço de fechamento para o(s) próximo(s) dia(s)</li>
                <li>Infra: Endpoint SageMaker em <code>us-east-2</code></li>
            </ul>

            <div class="rotas">
                <h2>📡 Rotas disponíveis</h2>

                <div class="rota">
                    <h3><code>GET /fechamento/hoje</code></h3>
                    <p>Retorna a previsão do fechamento para o <strong>próximo dia útil (D+0)</strong>.</p>
                </div>

                <div class="rota">
                    <h3><code>GET /fechamento/proximos</code></h3>
                    <p>Retorna a sequência de previsões para os <strong>próximos 5 dias úteis (D+1 a D+5)</strong>.</p>
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# --- Rotas da API ---
@app.get("/fechamento/hoje", summary="Previsão do próximo fechamento (D+0)")
def previsao_hoje():
    try:
        sequencia = get_ultimos_60_fechamentos()
        runtime = boto3.client("sagemaker-runtime", region_name=REGIAO)
        pred = prever_um_dia(sequencia, runtime)
        return {"previsao_hoje": pred}
    except Exception as e:
        return {"erro": str(e)}

@app.get("/fechamento/proximos", summary="Previsão dos próximos 5 fechamentos (D+1 a D+5)")
def previsao_proximos():
    try:
        previsoes = prever_proximos_dias(5)
        return {"previsoes_proximos_5_dias": previsoes}
    except Exception as e:
        return {"erro": str(e)}