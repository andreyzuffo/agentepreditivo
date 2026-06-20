import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
import pandas as pd
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "melhor_modelo_boston.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler_boston.pkl"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = (
    "Voce e um analista de dados esportivos. "
    "Explique resultados de classificacao de genero baseados em idade e tempo de prova. "
    "Seja objetivo, use apenas os dados fornecidos, nao invente informacoes e responda em no maximo 4 frases. "
    "O tempo de prova ja vem formatado em horas/minutos/segundos no prompt — use exatamente esse formato na resposta, nunca cite o tempo em segundos."
)

CHAT_SYSTEM_PROMPT = (
    "Voce e um assistente especializado exclusivamente na Maratona de Boston 2019 e no classificador de genero deste projeto (que usa idade e tempo de prova para prever o genero do corredor com um modelo MLP). "
    "Voce pode explicar o dataset, o modelo, as metricas obtidas e estatisticas da prova de 2019. "
    " Se o usuario perguntar sobre qualquer assunto fora desse escopo, responda educadamente que voce foi treinado apenas para falar sobre a Maratona de Boston e o classificador deste projeto, e nao tente responder a pergunta fora do escopo. Seja conciso nas respostas."
)

app = FastAPI(title="Agente Preditivo da Maratona de Boston: Gênero")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Nao foi encontrado o modelo/scaler em {MODEL_PATH} e {SCALER_PATH}."
    ) from exc

genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

if genai_client is None:
    print("\nGEMINI_API_KEY nao encontrada.\n")

class InputData(BaseModel):
    age: float = Field(..., gt=0, le=120, description="Idade do corredor")
    result_sec: float = Field(..., gt=0, description="Tempo de prova em segundos")


class ChatMessage(BaseModel):
    role: str 
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


def run_prediction(data: InputData) -> dict:  
    X_df = pd.DataFrame([[data.age, data.result_sec]], columns=["Age", "Result_sec"])
    X = scaler.transform(X_df)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {
        "prediction": "Masculino" if pred == 1 else "Feminino",
        "prob_male": float(proba[1]),
        "prob_female": float(proba[0]),
    }


def FormatHora(seconds: float) -> str:
    """Converte segundos em uma string 'Xh Ymin Zs' (ex.: 12000 -> '3h 20min 0s')."""
    total = int(round(seconds))
    h, resto = divmod(total, 3600)
    m, s = divmod(resto, 60)
    return f"{h}h {m}min {s}s"


def build_explanation_prompt(data: InputData, pred_result: dict) -> str:
    TempoFormatado = FormatHora(data.result_sec)
    return (
        f"Uma pessoa de {data.age} anos completou a Maratona de Boston em "
        f"{TempoFormatado} (tempo ja convertido de segundos para "
        f"horas/minutos/segundos).\n"
        f"O modelo classificou o genero como: {pred_result['prediction']}\n"
        f"(Probabilidade Masculino: {pred_result['prob_male']:.1%}, "
        f"Feminino: {pred_result['prob_female']:.1%})\n"
        "Explique esse resultado em linguagem natural, citando o tempo de "
        "prova no formato horas/minutos/segundos (nao em segundos)."
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    try:
        return run_prediction(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na predicao: {exc}") from exc


@app.post("/explain")
def explain(data: InputData):
    pred_result = run_prediction(data)

    if genai_client is None:
        raise HTTPException(
            status_code=503,
            detail=("GEMINI_API_KEY nao configurada."),
        )

    prompt = build_explanation_prompt(data, pred_result)

    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        explanation_text = response.text
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Erro ao chamar o Gemini: {exc}") from exc

    if not explanation_text:
        raise HTTPException(
            status_code=502,
            detail="O Gemini nao retornou texto (resposta vazia). Tente novamente.",
        )

    return {"prediction": pred_result, "explanation": explanation_text}


@app.post("/chat")
def chat(req: ChatRequest):
    if genai_client is None:
        raise HTTPException(
            status_code=503,
            detail=("GEMINI_API_KEY nao configurada"),
        )

    historico = req.messages[-20:]
    contents = [
        types.Content(role=m.role, parts=[types.Part(text=m.content)])
        for m in historico
    ]

    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=CHAT_SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=600,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        reply_text = response.text
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Erro ao chamar o Gemini: {exc}") from exc

    if not reply_text:
        raise HTTPException(
            status_code=502,
            detail="O Gemini nao retornou texto (resposta vazia). Tente novamente.",
        )

    return {"reply": reply_text}