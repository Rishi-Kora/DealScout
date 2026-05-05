"""Mock HTTP server for M7's real-API integration lesson.

Run this alongside agents.py so rc_lookup can make real HTTP calls instead of
in-memory dict lookups. This server intentionally misbehaves (timeouts, 500s,
auth failures, latency variance) to teach what real APIs feel like.

# Run

    uvicorn mock_server:app --port 8080

(In a separate terminal from where you run agents.py.)

# Configuration via env vars

    MOCK_API_KEY          required header value (default 'dev-secret-key')
    MOCK_FAILURE_RATE     probability of a random 500 (0.0 to 1.0, default 0.0)
    MOCK_LATENCY_MIN_S    min latency in seconds (default 1.0)
    MOCK_LATENCY_MAX_S    max latency in seconds (default 3.0)

# Surgical failure injection

    GET /rc-lookup/{reg_no}?simulate=timeout   sleeps 30s — forces client timeout
    GET /rc-lookup/{reg_no}?simulate=500       returns 500
"""

import asyncio
import logging
import os
import random

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from tools import RC_DATA

logging.getLogger("uvicorn.access").disabled = True

API_KEY = os.getenv("MOCK_API_KEY", "dev-secret-key")
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0.0"))
LATENCY_MIN = float(os.getenv("MOCK_LATENCY_MIN_S", "1.0"))
LATENCY_MAX = float(os.getenv("MOCK_LATENCY_MAX_S", "3.0"))

app = FastAPI(title="DealScout Mock RC API", version="1.0")


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    """Real APIs reject requests without valid auth. Health is open."""
    if request.url.path == "/health":
        return await call_next(request)
    if request.headers.get("X-API-Key") != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid X-API-Key header"},
        )
    return await call_next(request)


@app.get("/health")
async def health():
    """Liveness check — open to all, no auth required."""
    return {"status": "ok"}


@app.get("/rc-lookup/{reg_no}")
async def rc_lookup(reg_no: str, simulate: str | None = None):
    """Fetch RC details for a registration number.

    Query params:
        simulate=timeout — sleep 30 seconds (forces client timeout)
        simulate=500     — force a 500 error
    """
    # Surgical failure injection via query param.
    if simulate == "timeout":
        await asyncio.sleep(30)
    if simulate == "500":
        raise HTTPException(status_code=500, detail="Simulated server error")

    # Random server error based on configured failure rate.
    if FAILURE_RATE > 0 and random.random() < FAILURE_RATE:
        raise HTTPException(
            status_code=500,
            detail=f"Random server error (failure rate={FAILURE_RATE})",
        )

    # Realistic latency.
    await asyncio.sleep(random.uniform(LATENCY_MIN, LATENCY_MAX))

    # Data lookup.
    if reg_no not in RC_DATA:
        raise HTTPException(
            status_code=404, detail=f"No RC record indexed for {reg_no}"
        )

    return {"status": "ok", "reg_no": reg_no, **RC_DATA[reg_no]}
