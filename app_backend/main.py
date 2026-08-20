"""Local inference API for the Uganda mobile-money hybrid credit-risk model."""

from __future__ import annotations

import csv
import io
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from .inference import HybridRiskService, InputError

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("RISK_APP_DB", ROOT / "app_backend" / "risk_assessments.db"))
service: HybridRiskService | None = None

app = FastAPI(title="Mobile Money Credit Risk API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Assessment(BaseModel):
    id: int
    borrower_id: str
    created_at: str
    default_probability: float
    decision: str


def initialise_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                borrower_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                default_probability REAL NOT NULL,
                decision TEXT NOT NULL
            )"""
        )
        connection.commit()


@app.on_event("startup")
def startup() -> None:
    initialise_db()
    global service
    try:
        service = HybridRiskService(ROOT)
        service.load()
    except Exception as exc:  # don't crash the API if models or reference files are missing
        logging.getLogger("app_backend").warning("Model/service initialization failed at startup: %s", exc)
        service = None


@app.get("/health")
def health() -> dict:
    return {"status": "ready", "model_loaded": bool(service and service.is_loaded)}


@app.get("/template", response_class=Response)
def csv_template() -> Response:
    columns = ["borrower_id", "transaction_date", "transaction_amount", "transaction_type", "balance"]
    content = io.StringIO()
    writer = csv.writer(content)
    writer.writerow(columns)
    for month in range(1, 13):
        writer.writerow(["UG-001", f"2025-{month:02d}-15", 125000 + month * 1000, "received", 125000 + month * 1000])
    return Response(content=content.getvalue(), media_type="text/csv")


@app.post("/assess")
async def assess(
    transactions: Annotated[UploadFile, File(description="CSV containing one borrower’s transactions")],
    borrower_id: Annotated[str, Form()],
    loan_amount: Annotated[float, Form()],
    income_source: Annotated[str, Form()] = "Unknown",
    sacco_membership: Annotated[str, Form()] = "Unknown",
    location: Annotated[str, Form()] = "Unknown",
    preferred_network: Annotated[str, Form()] = "Unknown",
    preferred_channel: Annotated[str, Form()] = "Unknown",
) -> dict:
    if not transactions.filename or not transactions.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file.")
    raw = await transactions.read()
    if len(raw) > 5_000_000:
        raise HTTPException(413, "CSV exceeds the 5 MB limit.")
    if not service or not service.is_loaded:
        # Provide a mock response so the frontend flow can be tested when model files are missing.
        try:
            import pandas as _pd
            data_preview = _pd.read_csv(io.BytesIO(raw))
        except Exception as _err:
            raise HTTPException(422, "The uploaded file is not a readable CSV.") from _err
        if borrower_id.strip() not in data_preview.get("borrower_id", data_preview.get("borrower", [])).astype(str).values:
            raise HTTPException(422, "No rows match the supplied borrower ID.")
        # Minimal mock prediction
        probability = 0.1234
        decision = "Mock decision (model not available)"
        result = {
            "borrower_id": borrower_id.strip(),
            "default_probability": round(probability, 4),
            "decision": decision,
            "threshold": None,
            "model": "mock",
            "transactions_used": int(len(data_preview[data_preview["borrower_id"].astype(str) == borrower_id.strip()])),
            "period": {"from": None, "to": None},
            "notice": "Mock result — replace with real model files for production inference.",
        }
    else:
        try:
            result = service.predict_csv(
                raw,
                borrower_id=borrower_id.strip(),
                borrower_metadata={
                    "loan_amount": loan_amount,
                    "income_source": income_source,
                    "sacco_membership": sacco_membership,
                    "location": location,
                    "preferred_network": preferred_network,
                    "preferred_channel": preferred_channel,
                },
            )
        except InputError as error:
            raise HTTPException(422, str(error)) from error

    created_at = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(DB_PATH)) as connection:
        cursor = connection.execute(
            "INSERT INTO assessments (borrower_id, created_at, default_probability, decision) VALUES (?, ?, ?, ?)",
            (borrower_id.strip(), created_at, result["default_probability"], result["decision"]),
        )
        connection.commit()
        result["assessment_id"] = cursor.lastrowid
    return result


@app.post("/explain")
async def explain(
    transactions: Annotated[UploadFile, File(description="CSV containing one borrower's transactions")],
    borrower_id: Annotated[str, Form()],
    loan_amount: Annotated[float, Form()],
) -> dict:
    if not service or not service.is_loaded:
        raise HTTPException(503, "Model not available.")
    raw = await transactions.read()
    try:
        return service.explain_csv(raw, borrower_id.strip(), {"loan_amount": loan_amount})
    except (InputError, ValueError) as error:
        raise HTTPException(422, str(error)) from error


@app.get("/assessments", response_model=list[Assessment])
def assessments() -> list[Assessment]:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        rows = connection.execute(
            "SELECT id, borrower_id, created_at, default_probability, decision FROM assessments ORDER BY id DESC LIMIT 50"
        ).fetchall()
    return [Assessment(id=row[0], borrower_id=row[1], created_at=row[2], default_probability=row[3], decision=row[4]) for row in rows]
