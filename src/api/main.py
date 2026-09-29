"""Phase 13: FastAPI Application for Hybrid Network Intrusion Detection System.

Provides high-throughput REST endpoints for:
- Live network flow evaluation (Hybrid Decision Engine)
- Batch flow evaluation
- Local SHAP explainability for threat analysts
- Real-time simulated telemetry stream
- Research benchmark metrics & configuration
- Embedded Interactive SOC Dashboard
"""

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.api.schemas import (
    BatchFlowRequest,
    BatchPredictionResponse,
    ExplainResponse,
    NetworkFlow,
    PredictionResponse,
    SampleFlowItem,
)
from src.api.service import service

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models, preprocessor, and explainers at application startup."""
    print("[*] Starting up NIDS FastAPI Service...")
    service.load_models()
    yield
    print("[*] Shutting down NIDS FastAPI Service.")


app = FastAPI(
    title="Hybrid Network Intrusion Detection System API",
    description="Research-grade hybrid NIDS combining Multiclass Random Forest and Isolation Forest for unseen zero-day threat identification.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dashboards and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ThresholdUpdate(BaseModel):
    anomaly_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the SOC Interactive Dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "message": "NIDS API is online. Static dashboard index.html is being prepared.",
        "docs_url": "/docs"
    })


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint and runtime status."""
    return {
        "status": "healthy",
        "service": "NIDS Hybrid Engine",
        "version": "1.0.0",
        "models_loaded": service.is_loaded,
        "explainer_ready": service.explainer is not None,
        "anomaly_threshold": service.engine.anomaly_threshold if service.engine else 0.47821,
        "confidence_threshold": service.engine.confidence_threshold if service.engine else 0.65,
        "timestamp": time.time(),
    }


@app.get("/api/categories", tags=["Metadata"])
async def get_categories():
    """Return canonical attack categories and indexing."""
    return {
        "categories": service.metadata.get("canonical_attack_categories", []),
        "label_to_idx": service.metadata.get("label_to_idx", {}),
    }


@app.get("/api/sample-flows", response_model=List[SampleFlowItem], tags=["Data"])
async def get_sample_flows():
    """Return pre-configured realistic network flow presets for live interactive testing."""
    return service.get_sample_presets()


@app.get("/api/benchmarks", tags=["Evaluation"])
async def get_benchmarks():
    """Return comparative research metrics, confusion matrices, and unseen attack detection rates."""
    return service.benchmarks


@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_flow(flow: NetworkFlow):
    """Evaluate a single network flow with the Hybrid Decision Engine."""
    try:
        flow_dict = flow.model_dump()
        result = service.predict_flow(flow_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(request: BatchFlowRequest):
    """Evaluate a batch of network flows with vectorized inference."""
    try:
        flows_dicts = [f.model_dump() for f in request.flows]
        result = service.predict_batch(flows_dicts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@app.post("/api/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_flow(flow: NetworkFlow, top_n: int = Query(6, ge=1, le=15)):
    """Compute local SHAP feature attributions for a network flow."""
    try:
        flow_dict = flow.model_dump()
        result = service.explain_flow(flow_dict, top_n=top_n)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")


@app.get("/api/stream", tags=["Simulation"])
async def stream_live_packets(count: int = Query(8, ge=1, le=30)):
    """Simulate a live feed of evaluated network packets from the test distribution."""
    try:
        items = service.get_simulated_stream_batch(count=count)
        return {
            "count": len(items),
            "packets": items,
            "timestamp": time.time(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream simulation error: {str(e)}")


@app.post("/api/thresholds", tags=["Configuration"])
async def update_thresholds(config: ThresholdUpdate):
    """Dynamically adjust Hybrid Decision Engine thresholds in real time."""
    if not service.engine:
        raise HTTPException(status_code=503, detail="Decision engine not initialized")
    
    if config.anomaly_threshold is not None:
        service.engine.anomaly_threshold = float(config.anomaly_threshold)
    if config.confidence_threshold is not None:
        service.engine.confidence_threshold = float(config.confidence_threshold)

    return {
        "message": "Thresholds updated successfully",
        "anomaly_threshold": service.engine.anomaly_threshold,
        "confidence_threshold": service.engine.confidence_threshold,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
