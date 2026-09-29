# Phase 12: SHAP-based Explainability Report

**Methodology:** `shap.TreeExplainer` on the 10-Class Random Forest Model  
**Feature Space:** 65 Transformed Features (31 Numerical + 34 One-Hot Encoded)  

## 1. Global Feature Attribution Highlights
Based on global SHAP attributions across all network traffic categories:
1. **Flow Rates (`num_rate`, `num_sload`, `num_dload`):** Primary discriminators separating high-volume attacks (DoS, Exploits) from Normal traffic.
2. **TCP Handshake & Timing (`num_tcprtt`, `num_synack`, `num_dinpkt`):** Critical for identifying reconnaissance probes and interactive exploits.
3. **Payload Characteristics (`num_smean`, `num_dmean`, `num_sbytes`):** Essential for differentiating micro-payloads (Shellcode) from standard HTTP/DNS transactions.

*Global SHAP Summary Plot: [reports/figures/shap_summary_plot.png](file:///c:/Users/Administrator/OneDrive/Desktop/Network%20Intrusion%20detection%20of%20Networks/reports/figures/shap_summary_plot.png)*

## 2. Sample Local Explanations (Real Flow Instances)
### Example Prediction: `Shellcode` (Confidence: 96.8%)
| Rank | Feature Name | Flow Value (Normalized) | SHAP Attribution | Impact |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `cat_proto_udp` | 1.0 | `+0.1614` | increases risk / promotes class |
| 2 | `num_sbytes` | 0.0506 | `+0.1124` | increases risk / promotes class |
| 3 | `num_smean` | 4.0698 | `+0.1096` | increases risk / promotes class |
| 4 | `num_ct_src_dport_ltm` | 0.0 | `+0.0675` | increases risk / promotes class |
| 5 | `num_ct_dst_sport_ltm` | 0.0 | `+0.0445` | increases risk / promotes class |

### Example Prediction: `Backdoor` (Confidence: 27.2%)
| Rank | Feature Name | Flow Value (Normalized) | SHAP Attribution | Impact |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `cat_proto_udp` | 0.0 | `+0.0689` | increases risk / promotes class |
| 2 | `cat_proto_sctp` | 1.0 | `+0.0200` | increases risk / promotes class |
| 3 | `num_ct_src_dport_ltm` | 0.0 | `+0.0184` | increases risk / promotes class |
| 4 | `num_ct_dst_sport_ltm` | 0.0 | `+0.0145` | increases risk / promotes class |
| 5 | `num_rate` | 0.8633 | `-0.0114` | decreases risk / suppresses class |

### Example Prediction: `Backdoor` (Confidence: 49.9%)
| Rank | Feature Name | Flow Value (Normalized) | SHAP Attribution | Impact |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `cat_proto_udp` | 0.0 | `+0.0505` | increases risk / promotes class |
| 2 | `num_sjit` | 0.0208 | `+0.0344` | increases risk / promotes class |
| 3 | `num_sload` | -0.0098 | `+0.0238` | increases risk / promotes class |
| 4 | `num_sbytes` | 0.6518 | `+0.0223` | increases risk / promotes class |
| 5 | `num_sinpkt` | 0.8748 | `+0.0193` | increases risk / promotes class |

### Example Prediction: `Generic` (Confidence: 100.0%)
| Rank | Feature Name | Flow Value (Normalized) | SHAP Attribution | Impact |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `cat_service_dns` | 1.0 | `+0.1814` | increases risk / promotes class |
| 2 | `num_sbytes` | -0.2423 | `+0.1264` | increases risk / promotes class |
| 3 | `num_ct_src_dport_ltm` | 3.75 | `+0.1042` | increases risk / promotes class |
| 4 | `num_ct_dst_sport_ltm` | 3.5 | `+0.1025` | increases risk / promotes class |
| 5 | `cat_service_-` | 0.0 | `+0.0876` | increases risk / promotes class |


## 3. Integration with Backend & Dashboard
- The `explain_flow()` method provides real-time per-flow attribution.
- This feeds directly into the FastAPI `/detect` and `/predict` endpoints to supply the React dashboard with live feature contribution graphs.
