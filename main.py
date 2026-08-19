import os

import dspy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


# -----------------------------
# Configuration
# -----------------------------

SERVER_API_KEY = os.getenv("DSPY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured")

if not SERVER_API_KEY:
    raise RuntimeError("DSPY_API_KEY is not configured")


# -----------------------------
# DSPy + Gemini
# -----------------------------

lm = dspy.LM(
    "gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0,
)

dspy.configure(lm=lm)


predictor = dspy.Predict(
    "text -> answer"
)


# -----------------------------
# FastAPI
# -----------------------------

app = FastAPI(
    title="AptlyStar DSPy Server",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str | None = None
    input: str | None = None


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "DSPy",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/predict")
def predict(
    request: PredictRequest,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):
    # Accept either:
    # X-API-Key: <key>
    # OR
    # Authorization: Bearer <key>

    supplied_key = x_api_key

    if not supplied_key and authorization:
        if authorization.startswith("Bearer "):
            supplied_key = authorization[7:]

    if supplied_key != SERVER_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    text = request.text or request.input

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Input text is required",
        )

    result = predictor(text=text)

    answer = str(result.answer)

    return {
        "answer": answer,
        "output": answer,
        "status": "success",
    }
