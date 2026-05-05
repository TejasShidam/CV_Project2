"""ANPR Parking Intelligence Dashboard.

Streamlit UI that provides:
  - Training metrics visualisation (from JSONL logs)
  - Dataset health (annotation status breakdown)
  - Live inference (upload image → /predict API)
  - Parking occupancy donut chart
  - Plate log table with duplicate highlighting
  - API latency & model drift monitoring
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ANPR Parking Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;
                       letter-spacing: 0.1em; margin: 0; }
    .metric-card p  { color: #f8fafc; font-size: 2rem; font-weight: 700;
                       margin: 0.25rem 0 0; }

    .plate-tag {
        display: inline-block;
        background: #1d4ed8;
        color: #fff;
        border-radius: 6px;
        padding: 2px 8px;
        font-family: monospace;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .dup-tag { background: #dc2626; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — API settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/000000/car.png",
        width=60,
    )
    st.title("ANPR Dashboard")
    st.caption("CV + CNN + Transformer System")
    st.divider()

    api_url = st.text_input("FastAPI Base URL", value="http://localhost:8000")
    st.divider()
    st.markdown("**Navigation**")
    page = st.radio(
        "Go to",
        ["📊 Training Metrics", "🖼 Live Inference", "🅿️ Parking Analytics", "📡 Monitoring"],
        label_visibility="collapsed",
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
METRICS_PATH = Path("artifacts/logs/train_metrics.jsonl")
METADATA_PATH = Path("artifacts/metadata.csv")
PLATE_LOG_PATH = Path("artifacts/plate_log.csv")


def _load_metrics() -> pd.DataFrame | None:
    if not METRICS_PATH.exists():
        return None
        
    rows = []
    for line in METRICS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
            
    return pd.DataFrame(rows) if rows else None


def _api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{api_url}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.warning(f"API unavailable: {exc}")
        return None


def _card(label: str, value: str) -> str:
    return (
        f'<div class="metric-card"><h3>{label}</h3><p>{value}</p></div>'
    )


# ===========================================================================
# PAGE: Training Metrics
# ===========================================================================
if page == "📊 Training Metrics":
    st.header("📊 Training Metrics")

    df = _load_metrics()

    if df is None or df.empty:
        st.info("No training logs found. Run `python scripts/train.py` first.")
    else:
        # KPI row
        last = df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(_card("Epochs Trained", str(int(last["epoch"]))), unsafe_allow_html=True)
        with c2:
            st.markdown(_card("Best Full Acc", f"{last['best_val_full_acc']:.2%}"), unsafe_allow_html=True)
        with c3:
            st.markdown(_card("Last Char Acc", f"{last['val_char_acc']:.2%}"), unsafe_allow_html=True)
        with c4:
            lr_val = last.get("lr", "—")
            st.markdown(_card("Last LR", f"{lr_val:.2e}" if isinstance(lr_val, float) else str(lr_val)), unsafe_allow_html=True)

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Accuracy over Epochs")
            fig = px.line(
                df,
                x="epoch",
                y=["val_char_acc", "val_full_acc", "best_val_full_acc"],
                markers=True,
                labels={"value": "Accuracy", "epoch": "Epoch", "variable": "Metric"},
                color_discrete_sequence=["#60a5fa", "#34d399", "#f472b6"],
                template="plotly_dark",
            )
            fig.update_layout(legend_title_text="", margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Training Loss")
            fig2 = px.area(
                df,
                x="epoch",
                y="train_loss",
                labels={"train_loss": "CE Loss", "epoch": "Epoch"},
                color_discrete_sequence=["#f87171"],
                template="plotly_dark",
            )
            fig2.update_layout(margin=dict(t=20))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Full Log")
        st.dataframe(df.sort_values("epoch", ascending=False), use_container_width=True, height=280)

    st.divider()
    st.subheader("Dataset Health")
    if METADATA_PATH.exists():
        meta = pd.read_csv(METADATA_PATH)
        counts = meta["status"].value_counts().reset_index()
        counts.columns = ["status", "count"]
        fig3 = px.bar(
            counts,
            x="status",
            y="count",
            color="status",
            text="count",
            color_discrete_sequence=["#34d399", "#f87171", "#fbbf24"],
            template="plotly_dark",
        )
        fig3.update_layout(showlegend=False, margin=dict(t=20))
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(meta.head(30), use_container_width=True)
    else:
        st.info("Run `python scripts/prepare_dataset.py` to generate metadata.")


# ===========================================================================
# PAGE: Live Inference
# ===========================================================================
elif page == "🖼 Live Inference":
    st.header("🖼 Live Inference")
    st.markdown("Upload a vehicle image and the system will locate and read the number plate.")

    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Drop an image here",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", use_column_width=True)

    with col_result:
        if uploaded:
            if st.button("🔍 Run ANPR", type="primary", use_container_width=True):
                with st.spinner("Running inference…"):
                    try:
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG")
                        buf.seek(0)
                        resp = requests.post(
                            f"{api_url}/predict",
                            files={"file": ("image.jpg", buf, "image/jpeg")},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        data = resp.json()

                        st.success("Plate recognised!")
                        plate = data.get("plate_text", "—")
                        conf  = data.get("confidence", None)
                        zone  = data.get("zone", "—")

                        st.markdown(
                            f'<div style="text-align:center;padding:1rem;">'
                            f'<span class="plate-tag" style="font-size:2.5rem;padding:0.5rem 1.5rem;">'
                            f'{plate}</span></div>',
                            unsafe_allow_html=True,
                        )
                        m1, m2 = st.columns(2)
                        m1.metric("Confidence", f"{conf:.2%}" if conf else "—")
                        m2.metric("Zone", zone)

                        with st.expander("Raw API response"):
                            st.json(data)

                        # Append to plate log
                        log_row = {"plate": plate, "confidence": conf, "zone": zone}
                        if PLATE_LOG_PATH.exists():
                            log_df = pd.read_csv(PLATE_LOG_PATH)
                        else:
                            log_df = pd.DataFrame(columns=["plate", "confidence", "zone"])
                        log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
                        PLATE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                        log_df.to_csv(PLATE_LOG_PATH, index=False)

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "Cannot reach the API server. "
                            "Start it with `uvicorn anpr.api.main:app --reload` "
                            "then try again."
                        )
                    except Exception as exc:
                        st.error(f"Error: {exc}")
        else:
            st.info("Upload an image on the left to get started.")


# ===========================================================================
# PAGE: Parking Analytics
# ===========================================================================
elif page == "🅿️ Parking Analytics":
    st.header("🅿️ Parking Analytics")

    # --- Occupancy ---
    occ_data = _api_get("/analytics/occupancy")
    if occ_data and "occupancy" in occ_data:
        occ = occ_data["occupancy"]
        if occ:
            st.subheader("Zone Occupancy")
            df_occ = pd.DataFrame(list(occ.items()), columns=["Zone", "Count"])
            total = df_occ["Count"].sum()

            col_donut, col_table = st.columns([1, 1])
            with col_donut:
                fig_donut = px.pie(
                    df_occ,
                    names="Zone",
                    values="Count",
                    hole=0.55,
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    template="plotly_dark",
                )
                fig_donut.update_traces(textinfo="label+percent")
                fig_donut.update_layout(
                    showlegend=False,
                    annotations=[dict(text=f"<b>{total}</b><br>vehicles", x=0.5, y=0.5,
                                      font_size=16, showarrow=False, font_color="#f8fafc")],
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_table:
                st.dataframe(df_occ, use_container_width=True, height=250)
        else:
            st.info("No occupancy data yet — start sending predictions.")
    else:
        st.info("API unavailable or no occupancy data.")

    st.divider()

    # --- Plate log ---
    st.subheader("Plate Detection Log")
    if PLATE_LOG_PATH.exists():
        log_df = pd.read_csv(PLATE_LOG_PATH)

        # Flag duplicates
        dup_mask = log_df.duplicated(subset=["plate"], keep=False)

        def _style_row(row: pd.Series) -> list[str]:
            if dup_mask.iloc[row.name]:  # type: ignore[attr-defined]
                return ["background-color: #450a0a; color: #fca5a5"] * len(row)
            return [""] * len(row)

        styled = log_df.style.apply(_style_row, axis=1)  # type: ignore[arg-type]
        st.dataframe(styled, use_container_width=True, height=320)

        dup_count = int(dup_mask.sum())
        if dup_count:
            st.warning(f"{dup_count} rows flagged as duplicates (highlighted in red).")
    else:
        st.info("No plate log yet — run some predictions in Live Inference.")


# ===========================================================================
# PAGE: Monitoring
# ===========================================================================
elif page == "📡 Monitoring":
    st.header("📡 API & Model Monitoring")

    col_lat, col_drift = st.columns(2)

    # --- Latency ---
    with col_lat:
        st.subheader("API Latency")
        lat_data = _api_get("/analytics/latency")
        if lat_data:
            count  = int(lat_data.get("count", 0))
            mean_ms = lat_data.get("mean_ms", 0.0)
            p95_ms  = lat_data.get("p95_ms", 0.0)

            st.markdown(_card("Total Requests", str(count)), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            fig_lat = go.Figure(go.Bar(
                x=["Mean latency", "P95 latency"],
                y=[mean_ms, p95_ms],
                marker_color=["#60a5fa", "#f472b6"],
            ))
            fig_lat.update_layout(
                yaxis_title="ms",
                template="plotly_dark",
                margin=dict(t=20),
            )
            st.plotly_chart(fig_lat, use_container_width=True)
        else:
            st.info("No latency data yet.")

    # --- Drift ---
    with col_drift:
        st.subheader("Model Drift (JS Divergence)")
        drift_data = _api_get("/analytics/drift")
        if drift_data is not None:
            js = drift_data.get("js_divergence", 0.0)
            colour = "#34d399" if js < 0.1 else ("#fbbf24" if js < 0.25 else "#f87171")
            status  = "✅ Stable" if js < 0.1 else ("⚠️ Drifting" if js < 0.25 else "🔴 High Drift")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=js,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "JS Divergence", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 1], "tickwidth": 1},
                    "bar": {"color": colour},
                    "steps": [
                        {"range": [0, 0.1],  "color": "#14532d"},
                        {"range": [0.1, 0.25], "color": "#78350f"},
                        {"range": [0.25, 1],  "color": "#450a0a"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": 0.25,
                    },
                },
            ))
            fig_gauge.update_layout(template="plotly_dark", margin=dict(t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"**Status:** {status}")
        else:
            st.info("API unavailable — start the server to see drift data.")
