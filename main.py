import os

import dspy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


# ============================================
# Environment variables
# ============================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DSPY_API_KEY = os.getenv("DSPY_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured")

if not DSPY_API_KEY:
    raise RuntimeError("DSPY_API_KEY is not configured")


# ============================================
# Configure DSPy with Gemini 3.6 Flash
# ============================================

lm = dspy.LM(
    "gemini/gemini-3.6-flash",
    api_key=GEMINI_API_KEY,
)

dspy.configure(lm=lm)


# ============================================
# DSPy Prediction
# ============================================

predictor = dspy.Predict(
    "text -> answer"
)


# ============================================
# FastAPI
# ============================================

app = FastAPI(
    title="AptlyStar DSPy Server",
    version="1.0.0",
)


# ============================================
# Request model
# ============================================

class PredictRequest(BaseModel):
    text: str | None = None
    input: str | None = None


# ============================================
# Root
# ============================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AptlyStar DSPy Server",
    }


# ============================================
# Health
# ============================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ============================================
# Predict
# ============================================

@app.post("/predict")
def predict(
    request: PredictRequest,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):

    # ----------------------------------------
    # Authenticate
    # ----------------------------------------

    supplied_key = x_api_key

    if not supplied_key and authorization:
        if authorization.startswith("Bearer "):
            supplied_key = authorization[7:]

    if supplied_key != DSPY_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    # ----------------------------------------
    # Get input
    # ----------------------------------------

    text = request.text or request.input

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Input text is required",
        )

    # ----------------------------------------
    # Run DSPy
    # ----------------------------------------

    try:
        result = predictor(text=text)

        answer = str(result.answer)

        return {
            "answer": answer,
            "output": answer,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DSPy prediction failed: {str(e)}",
        )
