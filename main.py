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
# Configure DSPy + Gemini
# ============================================

lm = dspy.LM(
    "gemini/gemini-3.6-flash",
    api_key=GEMINI_API_KEY,
)

dspy.configure(lm=lm)


# ============================================
# DSPy Chain of Thought
# ============================================

cot = dspy.ChainOfThought(
    "question -> answer"
)


# ============================================
# FastAPI
# ============================================

app = FastAPI(
    title="AptlyStar DSPy Server",
    version="2.0.0",
)


# ============================================
# Request model
# ============================================

class PredictRequest(BaseModel):
    question: str | None = None
    text: str | None = None
    input: str | None = None


# ============================================
# Root
# ============================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AptlyStar DSPy Chain of Thought Server",
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
# Predict / Chain of Thought
# ============================================

@app.post("/predict")
def predict(
    request: PredictRequest,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
):

    # ----------------------------------------
    # Authentication
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
    # Get question
    # ----------------------------------------

    question = request.question or request.text or request.input

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question is required",
        )

    # ----------------------------------------
    # Run Chain of Thought
    # ----------------------------------------

    try:
        result = cot(question=question)

        answer = str(result.answer)
        reasoning = str(result.reasoning)

        return {
            "answer": answer,
            "reasoning": reasoning,
            "output": answer,
            "status": "success",
            "rawOutput": {
                "answer": answer,
                "reasoning": reasoning,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DSPy Chain of Thought failed: {str(e)}",
        )
