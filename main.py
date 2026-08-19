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
# ReAct Agent
# ============================================

react_agent = dspy.ReAct(
    signature="task -> answer",
    tools=[],
    max_iters=5,
)


# ============================================
# FastAPI
# ============================================

app = FastAPI(
    title="AptlyStar DSPy Server",
    version="3.0.0",
)


# ============================================
# Request model
# ============================================

class ReActRequest(BaseModel):
    task: str | None = None
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
        "service": "AptlyStar DSPy ReAct Server",
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
# ReAct endpoint
# ============================================

@app.post("/predict")
def predict(
    request: ReActRequest,
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
    # Get task
    # ----------------------------------------

    task = (
        request.task
        or request.question
        or request.text
        or request.input
    )

    if not task:
        raise HTTPException(
            status_code=400,
            detail="Task is required",
        )

    # ----------------------------------------
    # Run ReAct
    # ----------------------------------------

    try:
        result = react_agent(task=task)

        answer = str(result.answer)

        reasoning = getattr(result, "reasoning", "")
        trajectory = getattr(result, "trajectory", {})

        return {
            "answer": answer,
            "output": answer,
            "reasoning": str(reasoning),
            "trajectory": trajectory,
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"DSPy ReAct failed: {str(e)}",
        )
