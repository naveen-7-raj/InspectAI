"""InspectAI: Industrial Machine Health Monitoring Dashboard.

Streamlit application for simulated real-time machine telemetry, 1D CNN fault diagnosis,
and 1D Grad-CAM explainability via FastAPI REST and WebSocket endpoints.
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from websockets.sync.client import connect

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InspectAI.Dashboard")

# Supported CWRU Benchmark Recordings Catalog
CWRU_OPTIONS = {
    "97": "97 (Normal Baseline - 0 HP, 1796 RPM)",
    "98": "98 (Normal Baseline - 1 HP, 1772 RPM)",
    "99": "99 (Normal Baseline - 2 HP, 1750 RPM)",
    "100": "100 (Normal Baseline - 3 HP, 1730 RPM)",
    "118": "118 (Ball Fault - 0.007\" - 0 HP, 1796 RPM)",
    "185": "185 (Ball Fault - 0.014\" - 0 HP, 1796 RPM)",
    "222": "222 (Ball Fault - 0.021\" - 0 HP, 1796 RPM)",
    "105": "105 (Inner Race Fault - 0.007\" - 0 HP, 1796 RPM)",
    "169": "169 (Inner Race Fault - 0.014\" - 0 HP, 1796 RPM)",
    "209": "209 (Inner Race Fault - 0.021\" - 0 HP, 1796 RPM)",
    "130": "130 (Outer Race Fault - 0.007\" - 0 HP, 1796 RPM)",
    "197": "197 (Outer Race Fault - 0.014\" - 0 HP, 1796 RPM)",
    "234": "234 (Outer Race Fault - 0.021\" - 0 HP, 1796 RPM)",
}

CLASS_COLOR_MAP = {
    "Normal": "#2ecc71",
    "Ball Fault": "#e74c3c",
    "Inner Race Fault": "#f39c12",
    "Outer Race Fault": "#9b59b6",
}


# ==============================================================================
# Helper & Network Utilities (Exported for Testing)
# ==============================================================================

def fetch_backend_health(base_url: str, timeout: float = 2.0) -> Dict[str, Any]:
    """Query FastAPI backend health endpoint.

    Args:
        base_url: Base HTTP URL of the backend (e.g. 'http://localhost:8000').
        timeout: HTTP request timeout in seconds.

    Returns:
        Dict containing connection status, model loading status, and metadata.
    """
    clean_url = base_url.rstrip("/")
    endpoint = f"{clean_url}/api/v1/health"
    try:
        resp = requests.get(endpoint, timeout=timeout)
        if resp.status_code == 200:
            return {"connected": True, "data": resp.json(), "error": None}
        elif resp.status_code == 503:
            return {
                "connected": True,
                "data": resp.json(),
                "error": "Backend online but model artifacts are not loaded (503 Degraded).",
            }
        else:
            return {"connected": False, "data": None, "error": f"HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as exc:
        return {"connected": False, "data": None, "error": str(exc)}


def build_websocket_url(
    base_url: str,
    recording_id: str,
    replay_speed: float = 1.0,
    include_gradcam: bool = False,
    include_waveform: bool = True,
    inference_mode: Optional[str] = None,
    calibrate_first: Optional[bool] = None,
) -> str:
    """Construct full WebSocket stream URL from base HTTP URL."""
    clean_url = base_url.strip().rstrip("/")
    if clean_url.startswith("https://"):
        ws_base = "wss://" + clean_url[8:]
    elif clean_url.startswith("http://"):
        ws_base = "ws://" + clean_url[7:]
    elif clean_url.startswith("ws://") or clean_url.startswith("wss://"):
        ws_base = clean_url
    else:
        ws_base = "ws://" + clean_url

    gradcam_param = "true" if include_gradcam else "false"
    waveform_param = "true" if include_waveform else "false"
    clean_rec = recording_id.replace(".mat", "").strip()

    query_parts = [
        f"replay_speed={replay_speed}",
        f"include_gradcam={gradcam_param}",
        f"include_waveform={waveform_param}",
    ]
    if inference_mode is not None:
        query_parts.append(f"inference_mode={inference_mode}")
    if calibrate_first is not None:
        query_parts.append(f"calibrate_first={'true' if calibrate_first else 'false'}")

    return f"{ws_base}/api/v1/stream/{clean_rec}?" + "&".join(query_parts)


def parse_stream_message(raw_msg: Union[str, bytes, dict]) -> Dict[str, Any]:
    """Parse and validate incoming WebSocket JSON message."""
    if isinstance(raw_msg, (str, bytes)):
        try:
            data = json.loads(raw_msg)
        except json.JSONDecodeError as exc:
            return {"event": "error", "error_type": "JSONDecodeError", "detail": str(exc)}
    elif isinstance(raw_msg, dict):
        data = raw_msg
    else:
        return {"event": "error", "error_type": "InvalidMessageType", "detail": "Unrecognized message format."}

    return data


def update_prediction_history(
    history: List[Dict[str, Any]],
    new_entry: Dict[str, Any],
    max_length: int = 50,
) -> List[Dict[str, Any]]:
    """Append a prediction entry and cap history length to prevent memory leaks."""
    updated = [new_entry] + history
    return updated[:max_length]


# ==============================================================================
# Visualization Generators (Matplotlib Industrial Dark Styling)
# ==============================================================================

def render_waveform_figure(
    samples: np.ndarray,
    recording_id: str,
    sequence_number: int,
    sampling_rate_hz: int = 12000,
) -> plt.Figure:
    """Render 2,048-sample vibration acceleration waveform with dark industrial styling."""
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#141721")
    ax.set_facecolor("#1a1e2d")

    timestamps_ms = np.arange(len(samples)) * (1000.0 / sampling_rate_hz)

    ax.plot(timestamps_ms, samples, color="#00bcd4", linewidth=0.9, alpha=0.95, label="Drive-End Acceleration (g)")
    ax.axhline(0, color="#555f7a", linestyle="--", linewidth=0.7, alpha=0.8)

    ax.set_xlim(0, timestamps_ms[-1] if len(timestamps_ms) > 0 else 170.67)
    ax.set_xlabel("Time (milliseconds)", fontsize=9.5, fontweight="bold", color="#a6b0cf")
    ax.set_ylabel("Amplitude (g)", fontsize=9.5, fontweight="bold", color="#a6b0cf")
    ax.set_title(
        f"Authentic CWRU Vibration Waveform | Recording: {recording_id} | Window #{sequence_number:03d} (2,048 samples at 12 kHz)",
        fontsize=10.5,
        fontweight="bold",
        color="#e0e6ed",
        pad=8,
    )
    ax.tick_params(colors="#a6b0cf", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#2d3548")

    ax.grid(True, linestyle=":", alpha=0.35, color="#555f7a")
    ax.legend(loc="upper right", fontsize=8.5, facecolor="#141721", edgecolor="#2d3548", labelcolor="#e0e6ed")

    plt.tight_layout()
    return fig


def render_scores_figure(uncalibrated_scores: Dict[str, float]) -> plt.Figure:
    """Render horizontal bar chart of uncalibrated model softmax scores."""
    fig, ax = plt.subplots(figsize=(10, 2.4), facecolor="#141721")
    ax.set_facecolor("#1a1e2d")

    classes = list(uncalibrated_scores.keys())
    scores = [uncalibrated_scores.get(c, 0.0) for c in classes]
    colors = [CLASS_COLOR_MAP.get(c, "#3498db") for c in classes]

    y_pos = np.arange(len(classes))
    bars = ax.barh(y_pos, scores, color=colors, height=0.55, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=9.5, fontweight="bold", color="#e0e6ed")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Uncalibrated Model Softmax Score", fontsize=9.5, fontweight="bold", color="#a6b0cf")
    ax.set_title(
        "Diagnostic Category Scores (Model Output Sensitivity, Not Empirical Probabilities)",
        fontsize=10,
        fontweight="bold",
        color="#e0e6ed",
        pad=8,
    )
    ax.tick_params(colors="#a6b0cf", labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#2d3548")

    ax.grid(True, axis="x", linestyle=":", alpha=0.35, color="#555f7a")

    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(
            width + 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"{score:.1%}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#e0e6ed",
        )

    plt.tight_layout()
    return fig


def render_gradcam_figure(
    samples: np.ndarray,
    heatmap: np.ndarray,
    predicted_class: str,
    sampling_rate_hz: int = 12000,
) -> plt.Figure:
    """Render aligned 2-panel figure showing waveform and 1D Grad-CAM attribution curve."""
    fig, (ax_wave, ax_cam) = plt.subplots(
        2, 1, figsize=(10, 4.5), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]}, facecolor="#141721"
    )
    ax_wave.set_facecolor("#1a1e2d")
    ax_cam.set_facecolor("#1a1e2d")

    timestamps_ms = np.arange(len(samples)) * (1000.0 / sampling_rate_hz)

    # 1. Waveform
    ax_wave.plot(timestamps_ms, samples, color="#00bcd4", linewidth=0.8, alpha=0.9, label="Drive-End Signal (g)")
    ax_wave.axhline(0, color="#555f7a", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_wave.set_ylabel("Amplitude (g)", fontsize=9, fontweight="bold", color="#a6b0cf")
    ax_wave.set_title(
        f"1D Grad-CAM Explainability | Target Class: {predicted_class} (Layer: conv3)",
        fontsize=10.5,
        fontweight="bold",
        color="#e0e6ed",
    )
    ax_wave.tick_params(colors="#a6b0cf", labelsize=8)
    for spine in ax_wave.spines.values():
        spine.set_color("#2d3548")
    ax_wave.grid(True, linestyle=":", alpha=0.3, color="#555f7a")

    # 2. Grad-CAM Curve
    ax_cam.plot(timestamps_ms, heatmap, color="#e74c3c", linewidth=1.1, label="Grad-CAM Attribution [0.0 - 1.0]")
    ax_cam.fill_between(timestamps_ms, 0, heatmap, color="#e74c3c", alpha=0.25)
    ax_cam.axhline(0.6, color="#f39c12", linestyle="--", linewidth=0.8, label="Salience Threshold (0.6)")
    ax_cam.set_ylabel("Attribution", fontsize=9, fontweight="bold", color="#a6b0cf")
    ax_cam.set_xlabel("Time (milliseconds)", fontsize=9, fontweight="bold", color="#a6b0cf")
    ax_cam.set_ylim(-0.05, 1.05)
    ax_cam.tick_params(colors="#a6b0cf", labelsize=8)
    for spine in ax_cam.spines.values():
        spine.set_color("#2d3548")
    ax_cam.grid(True, linestyle=":", alpha=0.3, color="#555f7a")
    ax_cam.legend(loc="upper right", fontsize=8, facecolor="#141721", edgecolor="#2d3548", labelcolor="#e0e6ed")

    plt.tight_layout()
    return fig


# ==============================================================================
# Streamlit Dashboard User Interface
# ==============================================================================

def init_session_state() -> None:
    """Initialize persistent Streamlit session state variables."""
    if "is_streaming" not in st.session_state:
        st.session_state["is_streaming"] = False
    if "stop_requested" not in st.session_state:
        st.session_state["stop_requested"] = False
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    if "total_windows_streamed" not in st.session_state:
        st.session_state["total_windows_streamed"] = 0
    if "latest_prediction" not in st.session_state:
        st.session_state["latest_prediction"] = None


def apply_custom_styles() -> None:
    """Inject custom industrial dark theme CSS styling."""
    st.markdown(
        """
        <style>
        .main {
            background-color: #0b0e14;
            color: #e0e6ed;
        }
        div[data-testid="stSidebar"] {
            background-color: #11141d;
            border-right: 1px solid #1f2433;
        }
        .metric-card {
            background-color: #161b26;
            border: 1px solid #232a3b;
            border-radius: 6px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .metric-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #8c9ba5;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f0f4f8;
            margin-top: 4px;
            line-height: 1.25;
            word-break: break-word;
        }
        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            white-space: nowrap;
        }
        .badge-healthy {
            background-color: #1b382b;
            color: #2ecc71;
            border: 1px solid #2ecc71;
        }
        .badge-degraded {
            background-color: #42291a;
            color: #f39c12;
            border: 1px solid #f39c12;
        }
        .badge-offline {
            background-color: #3d1b1b;
            color: #e74c3c;
            border: 1px solid #e74c3c;
        }
        .scientific-notice {
            background-color: #131722;
            border-left: 3px solid #3498db;
            padding: 8px 12px;
            font-size: 0.8rem;
            color: #a6b0cf;
            margin: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="InspectAI – Industrial Health Monitor",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    apply_custom_styles()

    # --------------------------------------------------------------------------
    # Sidebar Controls
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### InspectAI Control Panel")
        st.caption("Explainable Edge Vibration Diagnosis")

        st.markdown("---")
        default_backend = os.getenv("BACKEND_URL", "http://localhost:8000")
        backend_url = st.text_input("Backend REST URL", value=default_backend)

        # Health Check
        health = fetch_backend_health(backend_url)
        if health["connected"] and health["data"] and health["data"].get("model_loaded"):
            st.markdown(
                f'<span class="status-badge badge-healthy">● Backend Connected: {health["data"].get("model_name")}</span>',
                unsafe_allow_html=True,
            )
        elif health["connected"]:
            st.markdown(
                '<span class="status-badge badge-degraded">▲ Backend Degraded (No Model)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="status-badge badge-offline">✕ Backend Offline</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Error: {health['error']}")

        st.markdown("---")
        st.markdown("#### Simulation Controls")

        rec_id_selected = st.selectbox(
            "Select CWRU Recording",
            options=list(CWRU_OPTIONS.keys()),
            format_func=lambda x: CWRU_OPTIONS[x],
            index=0,
            disabled=st.session_state["is_streaming"],
        )

        replay_speed = st.selectbox(
            "Replay Speed Multiplier",
            options=[0.5, 1.0, 2.0, 5.0],
            index=1,
            disabled=st.session_state["is_streaming"],
            format_func=lambda x: f"{x}× ({'Real-time' if x==1.0 else ('Slow' if x<1.0 else 'Fast')})",
        )

        enable_gradcam = st.checkbox(
            "Enable 1D Grad-CAM (Attribution Heatmap)",
            value=False,
            disabled=st.session_state["is_streaming"],
            help="Computes activation gradients on conv3 layer. Disabled by default for low CPU latency.",
        )

        st.markdown("#### Adaptive Inference & Calibration")
        inference_mode_sel = st.radio(
            "Inference Strategy",
            options=["adaptive", "baseline"],
            format_func=lambda x: "Adaptive (CORAL Alignment)" if x == "adaptive" else "Baseline (Frozen Release Model)",
            index=0,
            disabled=st.session_state["is_streaming"],
            help="Adaptive mode applies unsupervised covariance alignment for cross-rig domain transfer.",
        )

        calibrate_first = st.checkbox(
            "Online Calibration (Buffer initial windows)",
            value=True,
            disabled=(st.session_state["is_streaming"] or inference_mode_sel == "baseline"),
            help="Buffers initial unlabelled stream windows to compute target covariance before adapting predictions.",
        )

        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start_clicked = st.button(
                "Start Replay",
                use_container_width=True,
                type="primary",
                disabled=st.session_state["is_streaming"],
            )
        with col_btn2:
            stop_clicked = st.button(
                "Stop Replay",
                use_container_width=True,
                disabled=not st.session_state["is_streaming"],
            )

        if stop_clicked:
            st.session_state["stop_requested"] = True
            st.session_state["is_streaming"] = False

        st.markdown("---")
        st.markdown(
            """
            <div class="scientific-notice">
            <strong>Simulation Disclaimer:</strong><br>
            All signals originate from historical Case Western Reserve University (CWRU) laboratory benchmarks.
            No live physical machinery or sensor is connected.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # Main Header
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
            <div>
                <h2 style="margin: 0; padding: 0; color: #f0f4f8; font-weight: 700; letter-spacing: -0.02em;">
                    InspectAI <span style="font-size: 1.1rem; font-weight: 400; color: #8c9ba5;">| Explainable Industrial Machine Health Monitor</span>
                </h2>
            </div>
            <div>
                <span class="status-badge" style="background-color: #1e2433; color: #00bcd4; border: 1px solid #2d3548;">
                    Data Source: Simulated CWRU Replay
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # Dynamic KPI & Monitoring Dashboard Layout
    # --------------------------------------------------------------------------
    kpi_placeholder = st.empty()
    waveform_placeholder = st.empty()
    scores_placeholder = st.empty()
    xai_placeholder = st.empty()
    history_placeholder = st.empty()

    def render_idle_dashboard():
        """Render initial or paused dashboard state."""
        lat_pred = st.session_state["latest_prediction"]
        pred_label = f"{lat_pred['predicted_class']}" if lat_pred else "Awaiting Stream"
        pred_color = CLASS_COLOR_MAP.get(lat_pred["predicted_class"], "#8c9ba5") if lat_pred else "#8c9ba5"
        base_pred = lat_pred.get("baseline_predicted_class", "--") if lat_pred else "--"
        mode_label = lat_pred.get("inference_mode", "Baseline").title() if lat_pred else "Baseline"
        win_count = st.session_state["total_windows_streamed"]

        with kpi_placeholder.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-title">Active Recording</div><div class="metric-value">{rec_id_selected}.mat</div></div>',
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-title">Inference Mode</div><div class="metric-value">{mode_label}</div></div>',
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-title">Diagnostic State</div><div class="metric-value" style="color: {pred_color};">{pred_label}</div></div>',
                    unsafe_allow_html=True,
                )
            with col4:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-title">Baseline Comparison</div><div class="metric-value" style="color: #8c9ba5;">{base_pred}</div></div>',
                    unsafe_allow_html=True,
                )
            with col5:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-title">Windows Streamed</div><div class="metric-value">{win_count}</div></div>',
                    unsafe_allow_html=True,
                )

        if not st.session_state["is_streaming"] and not st.session_state["prediction_history"]:
            with waveform_placeholder.container():
                st.info("Click 'Start Replay' in the sidebar to stream authentic CWRU vibration recordings and live predictions.")

    render_idle_dashboard()

    # --------------------------------------------------------------------------
    # WebSocket Streaming Execution Loop
    # --------------------------------------------------------------------------
    if start_clicked:
        if not health["connected"] or not (health["data"] and health["data"].get("model_loaded")):
            st.error("Cannot start stream: FastAPI backend is unreachable or model is not loaded.")
            return

        st.session_state["is_streaming"] = True
        st.session_state["stop_requested"] = False
        st.session_state["prediction_history"] = []
        st.session_state["total_windows_streamed"] = 0

        ws_url = build_websocket_url(
            base_url=backend_url,
            recording_id=rec_id_selected,
            replay_speed=replay_speed,
            include_gradcam=enable_gradcam,
            include_waveform=True,
            inference_mode=inference_mode_sel,
            calibrate_first=calibrate_first,
        )

        logger.info(f"Connecting to WebSocket: {ws_url}")

        try:
            with connect(ws_url) as websocket:
                for raw_msg in websocket:
                    if st.session_state["stop_requested"]:
                        logger.info("Stopping WebSocket stream per user request.")
                        websocket.close()
                        break

                    msg = parse_stream_message(raw_msg)
                    event_type = msg.get("event")

                    if event_type == "error":
                        st.error(f"Stream Error ({msg.get('error_type')}): {msg.get('detail')}")
                        break

                    elif event_type == "stream_completed":
                        st.success(
                            f"Stream Completed: All {msg.get('total_windows_streamed')} windows from "
                            f"recording {msg.get('recording_id')} have been processed."
                        )
                        st.session_state["is_streaming"] = False
                        break

                    elif event_type == "prediction":
                        st.session_state["latest_prediction"] = msg
                        st.session_state["total_windows_streamed"] += 1

                        # Update History
                        history_entry = {
                            "Seq #": msg["sequence_number"],
                            "Start Time (s)": round(msg["window_start_seconds"], 3),
                            "Mode": msg.get("inference_mode", "baseline").upper(),
                            "Predicted Class": msg["predicted_class"],
                            "Baseline Prediction": msg.get("baseline_predicted_class", "--"),
                            "Confidence (Uncalibrated)": f"{msg['uncalibrated_scores'][msg['predicted_class']]:.2%}",
                            "Latency (ms)": f"{msg['inference_time_ms']:.2f}",
                        }
                        st.session_state["prediction_history"] = update_prediction_history(
                            st.session_state["prediction_history"], history_entry, max_length=50
                        )

                        # 1. Update KPI Row
                        pred_class = msg["predicted_class"]
                        pred_color = CLASS_COLOR_MAP.get(pred_class, "#8c9ba5")
                        mode_str = msg.get("inference_mode", "baseline").title()
                        base_class = msg.get("baseline_predicted_class", "--")

                        with kpi_placeholder.container():
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.markdown(
                                    f'<div class="metric-card"><div class="metric-title">Active Recording</div><div class="metric-value">{rec_id_selected}.mat</div></div>',
                                    unsafe_allow_html=True,
                                )
                            with col2:
                                st.markdown(
                                    f'<div class="metric-card"><div class="metric-title">Inference Mode</div><div class="metric-value">{mode_str}</div></div>',
                                    unsafe_allow_html=True,
                                )
                            with col3:
                                st.markdown(
                                    f'<div class="metric-card"><div class="metric-title">Diagnostic State</div><div class="metric-value" style="color: {pred_color};">{pred_class}</div></div>',
                                    unsafe_allow_html=True,
                                )
                            with col4:
                                st.markdown(
                                    f'<div class="metric-card"><div class="metric-title">Baseline Compare</div><div class="metric-value" style="color: #8c9ba5;">{base_class}</div></div>',
                                    unsafe_allow_html=True,
                                )
                            with col5:
                                st.markdown(
                                    f'<div class="metric-card"><div class="metric-title">Windows Streamed</div><div class="metric-value">{st.session_state["total_windows_streamed"]}</div></div>',
                                    unsafe_allow_html=True,
                                )

                            # Telemetry Warning if Fallback or Calibrating
                            if msg.get("inference_mode") == "calibrating":
                                st.warning("Calibration Active: Buffering initial unlabelled stream windows to compute covariance.")
                            elif msg.get("inference_mode") == "adaptive_fallback":
                                st.info("Adaptive Fallback: Domain adapter not calibrated; falling back to unadapted baseline.")

                        # 2. Update Waveform Visualization
                        if "waveform" in msg and msg["waveform"]:
                            samples = np.array(msg["waveform"], dtype=np.float32)
                            fig_wave = render_waveform_figure(
                                samples=samples,
                                recording_id=rec_id_selected,
                                sequence_number=msg["sequence_number"],
                                sampling_rate_hz=msg.get("sampling_rate_hz", 12000),
                            )
                            with waveform_placeholder.container():
                                st.pyplot(fig_wave, use_container_width=True)
                            plt.close(fig_wave)

                        # 3. Update Softmax Scores Bar Chart
                        if "uncalibrated_scores" in msg:
                            fig_scores = render_scores_figure(msg["uncalibrated_scores"])
                            with scores_placeholder.container():
                                st.pyplot(fig_scores, use_container_width=True)
                            plt.close(fig_scores)

                        # 4. Optional Grad-CAM Panel
                        if enable_gradcam and "gradcam_heatmap" in msg and msg["gradcam_heatmap"]:
                            heatmap = np.array(msg["gradcam_heatmap"], dtype=np.float32)
                            fig_cam = render_gradcam_figure(
                                samples=samples,
                                heatmap=heatmap,
                                predicted_class=pred_class,
                                sampling_rate_hz=msg.get("sampling_rate_hz", 12000),
                            )
                            with xai_placeholder.container():
                                st.markdown("#### Explainable AI (1D Grad-CAM Attribution)")
                                st.pyplot(fig_cam, use_container_width=True)
                                st.markdown(
                                    """
                                    <div class="scientific-notice">
                                    <strong>Scientific Caution:</strong> Grad-CAM highlights temporal feature gradient sensitivity 
                                    within layer <code>conv3</code>. It indicates mathematical model attribution, not physical proof 
                                    of defect dynamics or remaining useful life (RUL).
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            plt.close(fig_cam)

                        # 5. Update Recent History Table
                        with history_placeholder.container():
                            st.markdown("#### Recent Stream Telemetry")
                            df_hist = pd.DataFrame(st.session_state["prediction_history"])
                            st.dataframe(df_hist, use_container_width=True, hide_index=True)

        except Exception as exc:
            logger.exception(f"WebSocket streaming connection failed: {exc}")
            st.error(f"WebSocket connection closed or failed: {exc}")
        finally:
            st.session_state["is_streaming"] = False
            render_idle_dashboard()


if __name__ == "__main__":
    main()
