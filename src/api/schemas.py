"""Pydantic schemas for NIDS FastAPI Service."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NetworkFlow(BaseModel):
    """Raw network flow feature representation based on UNSW-NB15 dataset schema."""
    dur: float = Field(0.000011, description="Record total duration in seconds")
    proto: str = Field("udp", description="Transaction protocol (e.g. tcp, udp, arp, ospf)")
    service: str = Field("-", description="Destination service (e.g. http, dns, ftp, smtp, -)")
    state: str = Field("INT", description="Transaction state (e.g. FIN, INT, CON, REQ, RST)")
    spkts: int = Field(2, description="Source to destination packet count")
    dpkts: int = Field(0, description="Destination to source packet count")
    sbytes: int = Field(496, description="Source to destination transaction bytes")
    dbytes: int = Field(0, description="Destination to source transaction bytes")
    rate: float = Field(90909.09, description="Packets per second")
    sload: float = Field(180363632.0, description="Source bits per second")
    dload: float = Field(0.0, description="Destination bits per second")
    sloss: int = Field(0, description="Source packets retransmitted or dropped")
    dloss: int = Field(0, description="Destination packets retransmitted or dropped")
    sinpkt: float = Field(0.011, description="Source interpacket arrival time (msec)")
    dinpkt: float = Field(0.0, description="Destination interpacket arrival time (msec)")
    sjit: float = Field(0.0, description="Source jitter (msec)")
    djit: float = Field(0.0, description="Destination jitter (msec)")
    swin: int = Field(0, description="Source TCP window advertisement value")
    stcpb: int = Field(0, description="Source TCP base sequence number")
    dtcpb: int = Field(0, description="Destination TCP base sequence number")
    dwin: int = Field(0, description="Destination TCP window advertisement value")
    tcprtt: float = Field(0.0, description="TCP connection setup round-trip time")
    synack: float = Field(0.0, description="TCP SYN to SYN-ACK time")
    ackdat: float = Field(0.0, description="TCP SYN-ACK to ACK time")
    smean: int = Field(248, description="Mean packet size transmitted by source")
    dmean: int = Field(0, description="Mean packet size transmitted by destination")
    trans_depth: int = Field(0, description="Pipelined HTTP request depth")
    response_body_len: int = Field(0, description="Actual uncompressed content size of HTTP response")
    ct_src_dport_ltm: int = Field(1, description="Connections with same source and destination port in 100 records")
    ct_dst_sport_ltm: int = Field(1, description="Connections with same destination and source port in 100 records")
    is_ftp_login: int = Field(0, description="1 if FTP session is accessed by user/password, else 0")
    ct_ftp_cmd: int = Field(0, description="Count of commands attempted during FTP session")
    ct_flw_http_mthd: int = Field(0, description="Count of HTTP methods in flow")
    is_sm_ips_ports: int = Field(0, description="1 if source and destination IP/ports are identical, else 0")


class BatchFlowRequest(BaseModel):
    """Batch of network flows for evaluation."""
    flows: List[NetworkFlow]


class ShapFeatureImpact(BaseModel):
    """Individual feature impact from SHAP."""
    feature: str
    shap_value: float
    feature_value: float
    impact: str


class PredictionResponse(BaseModel):
    """Prediction output from the Hybrid Decision Engine."""
    decision: str = Field(..., description="NORMAL | KNOWN ATTACK | UNSEEN / ANOMALOUS")
    predicted_category: str = Field(..., description="Predicted multiclass attack type")
    confidence: float = Field(..., description="Supervised model confidence [0.0 - 1.0]")
    anomaly_score: float = Field(..., description="Isolation Forest anomaly score")
    anomaly_threshold: float = Field(..., description="Configured anomaly decision boundary")
    is_anomalous: bool = Field(..., description="True if anomaly score >= threshold")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    explanation: str = Field(..., description="Detailed AI decision rationale")
    inference_time_ms: float = Field(..., description="End-to-end evaluation latency in ms")


class BatchPredictionResponse(BaseModel):
    """Batch evaluation response."""
    total_evaluated: int
    decisions_summary: Dict[str, int]
    results: List[PredictionResponse]


class ExplainResponse(BaseModel):
    """Detailed SHAP feature attribution explanation."""
    prediction: PredictionResponse
    target_class: str
    top_features: List[ShapFeatureImpact]


class SampleFlowItem(BaseModel):
    """Curated representative network flow preset."""
    id: str
    name: str
    category: str
    expected_decision: str
    description: str
    flow: Dict[str, Any]
