"""
T231 — Explainable AI Audit Dashboard
======================================
Construction-site safety detection | YOLO object detection |
Test evaluation and false-positive/false-negative audit
with Grad-CAM, Score-CAM, and Occlusion Sensitivity explainability.
"""

import json
import traceback
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from explainability import YOLOGradCAM, score_cam, occlusion_sensitivity

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MODEL_PATH = ROOT / "models" / "best.pt"

st.set_page_config(
    page_title="T231 | Explainable AI Audit",
    page_icon="🦺",
    layout="wide",
)


# ── helper loaders ──────────────────────────────────────────────────────── #

@st.cache_data
def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def load_model(path):
    return YOLO(str(path))


def _image_png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ── data ----------------------------------------------------------------- #
metrics = load_json(ASSETS / "test_metrics.json")
metadata = load_json(ASSETS / "model_metadata.json")
condition_perf = load_json(ASSETS / "condition_performance.json")
audit = load_csv(ASSETS / "error_audit.csv")
class_summary = load_csv(ASSETS / "class_error_summary.csv")


# ── header ---------------------------------------------------------------- #
st.title("🦺 T231: Explainable AI Audit Dashboard")
st.caption(
    "Construction-site safety detection | YOLO object detection | "
    "Test evaluation and false-positive/false-negative audit"
)

if not MODEL_PATH.exists():
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.stop()


# ── sidebar --------------------------------------------------------------- #
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Condition Analysis",
        "Image Detection",
        "Error Audit",
        "Model & Limitations",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("**Model:** YOLOv8n")
st.sidebar.write("**Dataset test split:** 82 images")
st.sidebar.write("**Audit matching IoU:** 0.50")


# ======================================================================== #
#  PAGE: Overview
# ======================================================================== #
if page == "Overview":
    st.header("Model performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", "85.76%")
    c2.metric("Recall", "69.53%")
    c3.metric("mAP@0.50", "74.94%")
    c4.metric("mAP@0.50:0.95", "45.20%")

    st.subheader("Evaluation summary")
    a, b, c = st.columns(3)
    a.metric("Test images", "82")
    b.metric("Ground-truth objects", "760")
    b.caption("Annotated objects in test split")
    c.metric("Checkpoint size", "5.96 MB")

    st.info(
        "These metrics are from the held-out test split. "
        "They measure object detection performance, not a guarantee "
        "that every safety violation will be detected."
    )

    st.subheader("Class-wise error summary")
    if not class_summary.empty:
        st.dataframe(class_summary, use_container_width=True)
        numeric_cols = [
            col for col in ["FP", "FN"]
            if col in class_summary.columns
        ]
        if "class" in class_summary.columns and numeric_cols:
            fig = px.bar(
                class_summary,
                x="class",
                y=numeric_cols,
                barmode="group",
                title="False Positives vs False Negatives by Class",
                color_discrete_sequence=["#ef4444", "#f59e0b"],
                labels={"value": "Count", "variable": "Error Type", "class": "Class"},
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("class_error_summary.csv is missing or empty.")

    st.subheader("Audit totals")
    fp_count = 87
    fn_count = 204
    x, y = st.columns(2)
    x.metric("False-positive records", fp_count)
    y.metric("False-negative records", fn_count)
    st.caption(
        "Audit counts use confidence >= 0.25 and same-class IoU >= 0.50 "
        "matching. Counts represent unmatched detections/objects, not "
        "necessarily the number of unique images."
    )


# ======================================================================== #
#  PAGE: Condition Analysis  (Feature 1 — Polished)
# ======================================================================== #
elif page == "Condition Analysis":
    st.header("📊 Performance Under Different Conditions")
    st.write(
        "This section shows how model performance varies across **lighting**, "
        "**weather**, and **crowd density** conditions.  Construction sites face "
        "widely varying environments; understanding these limitations is "
        "critical for responsible deployment."
    )

    if not condition_perf:
        st.error("Condition performance data not found (assets/condition_performance.json).")
    else:
        # ── tabs for each condition category ─────────────────────────────── #
        tab_light, tab_weather, tab_crowd, tab_compare = st.tabs(
            ["💡 Lighting", "🌦️ Weather", "👷 Crowd Density", "📈 Cross-Condition Comparison"]
        )

        def _show_condition_tab(key, title, emoji, color_map):
            data = condition_perf.get(key, {})
            conditions = data.get("conditions", [])
            precision  = data.get("precision", [])
            recall     = data.get("recall", [])
            mAP50      = data.get("mAP50", [])
            notes      = data.get("notes", "")

            if not conditions:
                st.warning(f"No {title.lower()} data available.")
                return

            df = pd.DataFrame({
                "Condition": conditions,
                "Precision (%)": precision,
                "Recall (%)": recall,
                "mAP@0.50 (%)": mAP50,
            })

            st.subheader(f"{emoji} {title} conditions")

            # Metric cards for best / worst
            best_idx = int(np.argmax(mAP50))
            worst_idx = int(np.argmin(mAP50))
            m1, m2, m3 = st.columns(3)
            m1.metric(
                f"Best – {conditions[best_idx]}",
                f"{mAP50[best_idx]:.1f}% mAP50",
            )
            m2.metric(
                f"Worst – {conditions[worst_idx]}",
                f"{mAP50[worst_idx]:.1f}% mAP50",
                delta=f"{mAP50[worst_idx] - mAP50[best_idx]:+.1f}%",
                delta_color="inverse",
            )
            m3.metric(
                "Performance gap",
                f"{mAP50[best_idx] - mAP50[worst_idx]:.1f}%",
            )

            # Interactive table with colour highlighting
            st.dataframe(
                df.style.format({
                    "Precision (%)": "{:.1f}",
                    "Recall (%)": "{:.1f}",
                    "mAP@0.50 (%)": "{:.1f}",
                }).highlight_max(
                    subset=["Precision (%)", "Recall (%)", "mAP@0.50 (%)"],
                    color="#1a472a",
                ).highlight_min(
                    subset=["Precision (%)", "Recall (%)", "mAP@0.50 (%)"],
                    color="#5c1a1a",
                ),
                use_container_width=True,
            )

            # Grouped bar chart with Plotly
            df_melted = df.melt(
                id_vars="Condition",
                value_vars=["Precision (%)", "Recall (%)", "mAP@0.50 (%)"],
                var_name="Metric",
                value_name="Score",
            )
            fig = px.bar(
                df_melted,
                x="Condition",
                y="Score",
                color="Metric",
                barmode="group",
                title=f"{title} — Precision / Recall / mAP@0.50",
                color_discrete_sequence=color_map,
                text="Score",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                yaxis_range=[0, 105],
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Radar chart
            fig_radar = go.Figure()
            for i, cond in enumerate(conditions):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[precision[i], recall[i], mAP50[i]],
                    theta=["Precision", "Recall", "mAP@0.50"],
                    fill="toself",
                    name=cond,
                    opacity=0.6,
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100]),
                    bgcolor="rgba(0,0,0,0)",
                ),
                title=f"{title} — Radar Comparison",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e5e7eb",
                showlegend=True,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            if notes:
                st.info(f"**Note:** {notes}")

        with tab_light:
            _show_condition_tab(
                "lighting", "Lighting", "💡",
                ["#3b82f6", "#10b981", "#f59e0b"],
            )

        with tab_weather:
            _show_condition_tab(
                "weather", "Weather", "🌦️",
                ["#6366f1", "#14b8a6", "#f97316"],
            )

        with tab_crowd:
            _show_condition_tab(
                "crowd_density", "Crowd Density", "👷",
                ["#ec4899", "#22d3ee", "#eab308"],
            )

        with tab_compare:
            st.subheader("📈 Cross-Condition mAP@0.50 Comparison")
            st.write(
                "Side-by-side view of how mAP@0.50 degrades under adverse "
                "lighting, weather, and crowd conditions."
            )

            compare_rows = []
            for key, label in [
                ("lighting", "Lighting"),
                ("weather", "Weather"),
                ("crowd_density", "Crowd Density"),
            ]:
                data = condition_perf.get(key, {})
                for cond, val in zip(
                    data.get("conditions", []), data.get("mAP50", [])
                ):
                    compare_rows.append({
                        "Category": label,
                        "Condition": cond,
                        "mAP@0.50 (%)": val,
                    })

            if compare_rows:
                compare_df = pd.DataFrame(compare_rows)
                fig_comp = px.bar(
                    compare_df,
                    x="Condition",
                    y="mAP@0.50 (%)",
                    color="Category",
                    barmode="group",
                    title="mAP@0.50 Across All Condition Categories",
                    color_discrete_sequence=["#3b82f6", "#10b981", "#f97316"],
                    text="mAP@0.50 (%)",
                )
                fig_comp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_comp.update_layout(
                    yaxis_range=[0, 100],
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e5e7eb",
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                st.dataframe(compare_df, use_container_width=True)

        # ── Error audit breakdown by condition (from real CSV data) ──── #
        st.divider()
        st.subheader("🔍 Error Distribution by Condition (from audit data)")
        st.write(
            "These charts come from the **actual error audit CSV** and show "
            "how false positives and false negatives distribute across conditions."
        )

        if not audit.empty and "lighting" in audit.columns:
            cond_tabs = st.tabs(["By Lighting", "By Weather", "By Crowd Density"])

            for tab, col, lbl in zip(
                cond_tabs,
                ["lighting", "weather", "crowd_density"],
                ["Lighting", "Weather", "Crowd Density"],
            ):
                with tab:
                    if col in audit.columns and "error_type" in audit.columns:
                        breakdown = (
                            audit.groupby([col, "error_type"])
                            .size()
                            .reset_index(name="Count")
                        )
                        breakdown.rename(columns={col: lbl}, inplace=True)
                        fig_bd = px.bar(
                            breakdown,
                            x=lbl,
                            y="Count",
                            color="error_type",
                            barmode="group",
                            title=f"FP / FN Breakdown by {lbl}",
                            color_discrete_map={"FP": "#ef4444", "FN": "#f59e0b"},
                        )
                        fig_bd.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#e5e7eb",
                        )
                        st.plotly_chart(fig_bd, use_container_width=True)
                    else:
                        st.info(f"No '{col}' data in audit CSV.")

        # Summary
        st.divider()
        st.subheader("Key takeaways")
        st.markdown(
            """
            | Factor | Primary challenge | Recommendation |
            |--------|-------------------|----------------|
            | **Low-light / Night** | Recall drops to ~46% | Add night-time training images; consider IR cameras |
            | **Rain / Fog** | Both precision and recall degrade | Augment training with weather effects |
            | **High crowd density** | Occlusion causes many missed detections | Use higher-resolution inference; add overlapping-box training |
            """
        )


# ======================================================================== #
#  PAGE: Image Detection  (Features 2 & 3)
# ======================================================================== #
elif page == "Image Detection":
    st.header("Image Detection")
    st.write(
        "Detect construction-site safety objects and generate "
        "explainability heatmaps using **Grad-CAM**, **Score-CAM**, "
        "or **Occlusion Sensitivity**."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="t231_image_upload",
    )

    # ── environmental condition selectors (Feature 3 – CSV) ──────────── #
    st.subheader("Scene conditions (included in CSV)")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        lighting_cond = st.selectbox(
            "💡 Lighting condition",
            ["Bright / Daylight", "Overcast / Cloudy",
             "Low-light / Dusk", "Night / Artificial", "Unknown"],
            index=0,
        )
    with cc2:
        weather_cond = st.selectbox(
            "🌦️ Weather condition",
            ["Clear", "Cloudy / Overcast", "Rain",
             "Fog / Haze", "Unknown"],
            index=0,
        )
    with cc3:
        crowd_cond = st.selectbox(
            "👷 Crowd density",
            ["Low (1-3 people)", "Medium (4-8 people)",
             "High (9-15 people)", "Very High (16+ people)", "Unknown"],
            index=0,
        )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file).convert("RGB")

        if st.session_state.get("t231_upload_bytes") != image_bytes:
            for key in ["t231_result", "t231_heatmap", "t231_overlay",
                        "t231_image", "t231_explained_idx"]:
                st.session_state.pop(key, None)
            st.session_state["t231_upload_bytes"] = image_bytes

        st.image(image, caption="Uploaded image", use_container_width=True)

        c1, c2, c3 = st.columns(3)
        confidence = c1.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
        nms_iou = c2.slider("NMS IoU", 0.10, 0.90, 0.45, 0.05)
        image_size = c3.select_slider(
            "Inference image size",
            options=[320, 480, 640, 800, 960],
            value=640,
        )

        if st.button("Run Detection", type="primary"):
            try:
                model = load_model(MODEL_PATH)
                with st.spinner("Running YOLO detection..."):
                    result = model.predict(
                        source=image, conf=confidence, iou=nms_iou,
                        imgsz=image_size, verbose=False,
                    )[0]
                st.session_state["t231_result"] = result
                st.session_state["t231_image"] = image
                st.session_state.pop("t231_heatmap", None)
                st.session_state.pop("t231_overlay", None)
                st.session_state.pop("t231_explained_idx", None)
            except Exception as exc:
                st.error(f"Detection failed: {exc}")

        result = st.session_state.get("t231_result")
        saved_image = st.session_state.get("t231_image")

        if result is not None and saved_image is not None:
            st.subheader("Detection result")
            st.image(
                result.plot()[:, :, ::-1].copy(),
                caption="Predicted bounding boxes",
                use_container_width=True,
            )

            rows = []
            if result.boxes is not None:
                for idx, box in enumerate(result.boxes):
                    class_id = int(box.cls.item())
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    rows.append({
                        "Detection": idx + 1,
                        "Class": result.names[class_id],
                        "Class ID": class_id,
                        "Confidence": round(float(box.conf.item()), 4),
                        "X1": round(coords[0], 1),
                        "Y1": round(coords[1], 1),
                        "X2": round(coords[2], 1),
                        "Y2": round(coords[3], 1),
                        "Lighting": lighting_cond,
                        "Weather": weather_cond,
                        "Crowd Density": crowd_cond,
                    })

            st.metric("Objects detected", len(rows))

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                # ── CSV download NOW includes condition columns ─────── #
                st.download_button(
                    "📥 Download detections CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="t231_detections.csv",
                    mime="text/csv",
                )

                st.divider()

                # ── Explainability ──────────────────────────────── #
                st.subheader("🔍 Explain a Selected Detection")
                st.write(
                    "Choose a detected object and an XAI method.  "
                    "The heatmap highlights regions the model relied on."
                )

                selected_idx = st.selectbox(
                    "Choose an object",
                    options=list(range(len(rows))),
                    format_func=lambda i: (
                        f'{i + 1}. {rows[i]["Class"]} '
                        f'(confidence {rows[i]["Confidence"]:.3f})'
                    ),
                )

                xai_method = st.radio(
                    "Explainability method",
                    ["Grad-CAM", "Score-CAM", "Occlusion Sensitivity"],
                    horizontal=True,
                    help=(
                        "**Grad-CAM** — gradient-weighted class activation map "
                        "(fast, gradient-based).  "
                        "**Score-CAM** — gradient-free variant using channel "
                        "importance scoring.  "
                        "**Occlusion Sensitivity** — perturbs image regions "
                        "to measure confidence drops."
                    ),
                )

                if xai_method == "Occlusion Sensitivity":
                    grid = st.slider("Explanation grid resolution", 4, 8, 6)
                elif xai_method == "Score-CAM":
                    top_k = st.slider(
                        "Score-CAM channels (higher = slower but finer)",
                        8, 32, 16,
                    )

                if st.button("Generate Explanation Heatmap"):
                    try:
                        box = result.boxes[selected_idx]
                        target_cls = int(box.cls.item())
                        target_box = box.xyxy[0].cpu().numpy().tolist()
                        original_score = float(box.conf.item())
                        model_obj = load_model(MODEL_PATH)

                        if xai_method == "Grad-CAM":
                            with st.spinner("Computing Grad-CAM heatmap (uses PyTorch backward hooks on backbone Conv2d)..."):
                                gcam = YOLOGradCAM(model_obj)
                                overlay, heatmap = gcam.generate(
                                    image=saved_image,
                                    target_box=target_box,
                                    target_class_id=target_cls,
                                    imgsz=image_size,
                                )
                                gcam.remove_hooks()

                        elif xai_method == "Score-CAM":
                            with st.spinner(
                                f"Computing Score-CAM ({top_k} channels, gradient-free forward passes)..."
                            ):
                                overlay, heatmap = score_cam(
                                    image=saved_image,
                                    model=model_obj,
                                    target_class_id=target_cls,
                                    target_box=target_box,
                                    imgsz=image_size,
                                    conf=0.05,
                                    iou=nms_iou,
                                    top_k_channels=top_k,
                                )

                        else:  # Occlusion Sensitivity
                            with st.spinner(
                                f"Testing up to {grid * grid} masked images..."
                            ):
                                overlay, heatmap = occlusion_sensitivity(
                                    image=saved_image,
                                    model=model_obj,
                                    target_class_id=target_cls,
                                    target_box=target_box,
                                    original_score=original_score,
                                    conf=0.05,
                                    iou=nms_iou,
                                    grid=grid,
                                    imgsz=image_size,
                                )

                        st.session_state["t231_overlay"] = overlay
                        st.session_state["t231_heatmap"] = heatmap
                        st.session_state["t231_explained_idx"] = selected_idx
                        st.session_state["t231_xai_method"] = xai_method
                    except Exception as exc:
                        st.error(f"Explanation failed: {exc}")
                        st.code(traceback.format_exc())

                if (
                    st.session_state.get("t231_heatmap") is not None
                    and st.session_state.get("t231_explained_idx") == selected_idx
                ):
                    used_method = st.session_state.get("t231_xai_method", "?")
                    left, right = st.columns(2)
                    left.image(
                        st.session_state["t231_heatmap"],
                        caption=f"{used_method} heatmap",
                        use_container_width=True,
                    )
                    right.image(
                        st.session_state["t231_overlay"],
                        caption=f"{used_method} overlay",
                        use_container_width=True,
                    )
                    st.download_button(
                        f"Download {used_method} heatmap PNG",
                        data=_image_png_bytes(
                            st.session_state["t231_heatmap"]
                        ),
                        file_name=f"t231_{used_method.lower().replace('-','')}_heatmap.png",
                        mime="image/png",
                    )

                    # Method-specific explanation
                    if used_method == "Grad-CAM":
                        st.success(
                            "**Grad-CAM** uses gradients flowing into the last "
                            "convolutional layer of the YOLO backbone to produce "
                            "a coarse localisation map highlighting important "
                            "regions. Warmer colours indicate higher importance. "
                            "This is computed using `register_full_backward_hook` "
                            "on the backbone's final Conv2d layer."
                        )
                    elif used_method == "Score-CAM":
                        st.success(
                            "**Score-CAM** is gradient-free: each activation "
                            "channel from the backbone's last Conv2d is used as "
                            "a soft mask, and the change in detection confidence "
                            "scores the channel's importance. The weighted sum "
                            "produces the heatmap — no gradients are needed."
                        )
                    else:
                        st.info(
                            "**Occlusion Sensitivity** replaces grid patches "
                            "with local mean colour. Brighter regions had a "
                            "larger confidence drop when occluded."
                        )
            else:
                st.info(
                    "No objects detected.  Try a lower confidence threshold."
                )
    else:
        st.info("Upload an image to begin.")


# ======================================================================== #
#  PAGE: Error Audit  (Feature 3 — enriched CSV)
# ======================================================================== #
elif page == "Error Audit":
    st.header("False-positive and false-negative audit")

    if audit.empty:
        st.error("Audit data not found. Check assets/error_audit.csv.")
    else:
        st.write(f"Loaded **{len(audit)}** audit records.")

        # ── primary filters ──────────────────────────────────────────── #
        fc_a, fc_b = st.columns(2)
        with fc_a:
            if "error_type" in audit.columns:
                error_types = sorted(
                    audit["error_type"].dropna().astype(str).unique()
                )
                chosen_type = st.selectbox("Error type", ["All"] + error_types)
            else:
                chosen_type = "All"
                st.warning("The CSV has no 'error_type' column.")

        with fc_b:
            if "class" in audit.columns:
                classes = sorted(audit["class"].dropna().astype(str).unique())
                chosen_class = st.selectbox("Class", ["All"] + classes)
            else:
                chosen_class = "All"

        # ── condition filters ────────────────────────────────────────── #
        st.subheader("Filter by environmental condition")
        fc1, fc2, fc3 = st.columns(3)

        lighting_opts = ["All"]
        weather_opts  = ["All"]
        crowd_opts    = ["All"]

        if "lighting" in audit.columns:
            lighting_opts += sorted(audit["lighting"].dropna().astype(str).unique())
        if "weather" in audit.columns:
            weather_opts += sorted(audit["weather"].dropna().astype(str).unique())
        if "crowd_density" in audit.columns:
            crowd_opts += sorted(audit["crowd_density"].dropna().astype(str).unique())

        with fc1:
            filter_lighting = st.selectbox(
                "💡 Lighting", lighting_opts, key="audit_lighting",
            )
        with fc2:
            filter_weather = st.selectbox(
                "🌦️ Weather", weather_opts, key="audit_weather",
            )
        with fc3:
            filter_crowd = st.selectbox(
                "👷 Crowd density", crowd_opts, key="audit_crowd",
            )

        filtered = audit.copy()

        if chosen_type != "All" and "error_type" in filtered.columns:
            filtered = filtered[
                filtered["error_type"].astype(str) == chosen_type
            ]
        if chosen_class != "All" and "class" in filtered.columns:
            filtered = filtered[
                filtered["class"].astype(str) == chosen_class
            ]
        if filter_lighting != "All" and "lighting" in filtered.columns:
            filtered = filtered[
                filtered["lighting"].astype(str) == filter_lighting
            ]
        if filter_weather != "All" and "weather" in filtered.columns:
            filtered = filtered[
                filtered["weather"].astype(str) == filter_weather
            ]
        if filter_crowd != "All" and "crowd_density" in filtered.columns:
            filtered = filtered[
                filtered["crowd_density"].astype(str) == filter_crowd
            ]

        st.metric("Matching audit records", len(filtered))
        st.dataframe(filtered, use_container_width=True)

        # ── CSV download includes condition columns ──────────────────── #
        st.download_button(
            "📥 Download filtered audit CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="t231_filtered_audit.csv",
            mime="text/csv",
        )

        # ── Quick stats chart ────────────────────────────────────────── #
        if "error_type" in filtered.columns and len(filtered) > 0:
            st.subheader("Filtered audit summary")

            err_summary_cols = []
            for col_name, col_label in [
                ("lighting", "Lighting"),
                ("weather", "Weather"),
                ("crowd_density", "Crowd Density"),
            ]:
                if col_name in filtered.columns:
                    err_summary_cols.append((col_name, col_label))

            if err_summary_cols:
                chart_tabs = st.tabs([lbl for _, lbl in err_summary_cols])
                for tab, (col_name, col_label) in zip(chart_tabs, err_summary_cols):
                    with tab:
                        grp = (
                            filtered.groupby([col_name, "error_type"])
                            .size()
                            .reset_index(name="Count")
                        )
                        grp.rename(columns={col_name: col_label}, inplace=True)
                        fig = px.bar(
                            grp,
                            x=col_label,
                            y="Count",
                            color="error_type",
                            barmode="group",
                            title=f"FP/FN by {col_label} (filtered)",
                            color_discrete_map={"FP": "#ef4444", "FN": "#f59e0b"},
                        )
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#e5e7eb",
                        )
                        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Error image galleries")
    gallery_type = st.radio(
        "Choose gallery",
        ["False positives", "False negatives"],
        horizontal=True,
    )
    gallery_folder = (
        ASSETS / "false_positives"
        if gallery_type == "False positives"
        else ASSETS / "false_negatives"
    )

    if gallery_folder.exists():
        image_paths = sorted(
            p for p in gallery_folder.rglob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        )
        st.write(f"{len(image_paths)} images available")

        if image_paths:
            selected = st.selectbox(
                "Inspect an image",
                image_paths,
                format_func=lambda p: str(p.relative_to(gallery_folder)),
            )
            st.image(
                str(selected),
                caption=selected.name,
                use_container_width=True,
            )
        else:
            st.info("No supported image files found in this gallery.")
    else:
        st.warning(f"Gallery folder not found: {gallery_folder}")


# ======================================================================== #
#  PAGE: Model & Limitations  (Updated)
# ======================================================================== #
elif page == "Model & Limitations":
    st.header("Model information")

    st.write("**Architecture:** YOLOv8n")
    st.write("**Parameters:** 3,007,598")
    st.write("**GFLOPs:** 8.1")
    st.write("**Checkpoint:** models/best.pt")
    st.write("**Test images:** 82")
    st.write("**Test instances:** 760")

    st.subheader("Saved model metadata")
    st.json(metadata if metadata else {"status": "Metadata unavailable"})

    st.subheader("Saved test metrics")
    st.json(metrics if metrics else {"status": "Metrics unavailable"})

    st.subheader("Explainability methods implemented")
    st.markdown(
        """
        | Method | Type | How it works |
        |--------|------|-------------|
        | **Grad-CAM** | Gradient-based | Hooks into the last Conv2d of the YOLO backbone using `register_forward_hook` and `register_full_backward_hook`. Computes gradient-weighted channel activations → class activation map. |
        | **Score-CAM** | Gradient-free | Extracts activation maps from the last backbone Conv2d. Upsamples each channel as a soft mask, forward-passes each masked image, and weights channels by detection confidence. |
        | **Occlusion Sensitivity** | Perturbation-based | Systematically occludes grid patches with local mean colour, re-runs detection, and measures confidence drops per patch. |
        """
    )

    st.subheader("Important limitations")
    st.markdown(
        """
        - This is a research prototype, not a certified safety system.
        - A predicted PPE violation is not definitive proof of a violation.
        - Missed detections are possible; test recall is 69.53 %.
        - Performance degrades in low-light, rain/fog, and high-crowd
          conditions (see the **Condition Analysis** page).
        - Bounding boxes and confidence scores explain the model's
          output at a basic level; they are not causal explanations.
        - This dashboard implements **Grad-CAM** and **Score-CAM**
          for class-activation-map-based explainability, as well as
          **Occlusion Sensitivity** for perturbation-based analysis.
        - False-positive and false-negative counts depend on the chosen
          confidence threshold and matching IoU.
        """
    )


# ── footer ─────────────────────────────────────────────────────────────── #
st.markdown("---")
st.caption(
    "T231 research prototype | YOLO object detection and error auditing"
)
