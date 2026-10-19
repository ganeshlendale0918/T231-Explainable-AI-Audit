import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from explainability import occlusion_sensitivity

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MODEL_PATH = ROOT / "models" / "best.pt"

st.set_page_config(
    page_title="T231 | Explainable AI Audit",
    page_icon="🦺",
    layout="wide",
)

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

metrics = load_json(ASSETS / "test_metrics.json")
metadata = load_json(ASSETS / "model_metadata.json")
audit = load_csv(ASSETS / "error_audit.csv")
class_summary = load_csv(ASSETS / "class_error_summary.csv")

st.title("🦺 T231: Explainable AI Audit Dashboard")
st.caption(
    "Construction-site safety detection | YOLO object detection | "
    "Test evaluation and false-positive/false-negative audit"
)

if not MODEL_PATH.exists():
    st.error(f"Model checkpoint not found: {MODEL_PATH}")
    st.stop()

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Image Detection",
        "Error Audit",
        "Model & Limitations",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("**Model:** YOLOv8n")
st.sidebar.write("**Dataset test split:** 82 images")
st.sidebar.write("**Audit matching IoU:** 0.50")



def _image_png_bytes(image):
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

if page == "Overview":
    st.header("Model performance")

    # Values are based on the independently evaluated test set.
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
            chart_data = class_summary.set_index("class")[numeric_cols]
            st.bar_chart(chart_data)
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

elif page == "Image Detection":
    st.header("Image Detection")
    st.write("Detect construction-site safety objects and inspect a sensitivity heatmap.")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="t231_image_upload",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(uploaded_file).convert("RGB")

        if st.session_state.get("t231_upload_bytes") != image_bytes:
            for key in ["t231_result", "t231_heatmap", "t231_overlay",
                        "t231_image", "t231_explained_idx"]:
                st.session_state.pop(key, None)
            st.session_state["t231_upload_bytes"] = image_bytes

        st.image(image, caption="Uploaded image", width="stretch")

        c1, c2, c3 = st.columns(3)
        confidence = c1.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
        nms_iou = c2.slider("NMS IoU", 0.10, 0.90, 0.45, 0.05)
        image_size = c3.select_slider(
            "Inference image size", options=[320, 480, 640, 800, 960],
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
            st.image(result.plot()[:, :, ::-1].copy(),
                     caption="Predicted bounding boxes", width="stretch")

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
                        "X1": round(coords[0], 1), "Y1": round(coords[1], 1),
                        "X2": round(coords[2], 1), "Y2": round(coords[3], 1),
                    })

            st.metric("Objects detected", len(rows))

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch")
                st.download_button(
                    "Download detections CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="t231_detections.csv",
                    mime="text/csv",
                )

                st.divider()
                st.subheader("Explain a Selected Detection")
                selected_idx = st.selectbox(
                    "Choose an object",
                    options=list(range(len(rows))),
                    format_func=lambda i: (
                        f'{i + 1}. {rows[i]["Class"]} '
                        f'(confidence {rows[i]["Confidence"]:.3f})'
                    ),
                )
                grid = st.slider("Explanation grid resolution", 4, 8, 6)

                if st.button("Generate Explanation Heatmap"):
                    try:
                        box = result.boxes[selected_idx]
                        with st.spinner(
                            f"Testing up to {grid * grid} masked images..."
                        ):
                            overlay, heatmap = occlusion_sensitivity(
                                image=saved_image,
                                model=load_model(MODEL_PATH),
                                target_class_id=int(box.cls.item()),
                                target_box=box.xyxy[0].cpu().numpy().tolist(),
                                original_score=float(box.conf.item()),
                                conf=0.05, iou=nms_iou,
                                grid=grid, imgsz=image_size,
                            )
                        st.session_state["t231_overlay"] = overlay
                        st.session_state["t231_heatmap"] = heatmap
                        st.session_state["t231_explained_idx"] = selected_idx
                    except Exception as exc:
                        st.error(f"Explanation failed: {exc}")

                if (
                    st.session_state.get("t231_heatmap") is not None
                    and st.session_state.get("t231_explained_idx") == selected_idx
                ):
                    left, right = st.columns(2)
                    left.image(
                        st.session_state["t231_heatmap"],
                        caption="Normalized confidence-drop heatmap",
                        width="stretch",
                    )
                    right.image(
                        st.session_state["t231_overlay"],
                        caption="Heatmap overlay",
                        width="stretch",
                    )
                    st.download_button(
                        "Download heatmap PNG",
                        data=_image_png_bytes(st.session_state["t231_heatmap"]),
                        file_name="t231_sensitivity_heatmap.png",
                        mime="image/png",
                    )
                    st.info(
                        "Brighter regions indicate areas where masking caused "
                        "a larger confidence drop. This is a perturbation-based "
                        "sensitivity visualization, not Grad-CAM or proof of causality."
                    )
            else:
                st.info("No objects detected. Try a lower confidence threshold.")
    else:
        st.info("Upload an image to begin.")
    
elif page == "Error Audit":
    st.header("False-positive and false-negative audit")

    if audit.empty:
        st.error("Audit data not found. Check assets/error_audit.csv.")
    else:
        st.write(f"Loaded {len(audit)} audit records.")

        if "error_type" in audit.columns:
            error_types = sorted(
                audit["error_type"].dropna().astype(str).unique()
            )
            chosen_type = st.selectbox("Error type", ["All"] + error_types)
        else:
            chosen_type = "All"
            st.warning("The CSV has no 'error_type' column.")

        if "class" in audit.columns:
            classes = sorted(audit["class"].dropna().astype(str).unique())
            chosen_class = st.selectbox("Class", ["All"] + classes)
        else:
            chosen_class = "All"

        filtered = audit.copy()

        if chosen_type != "All" and "error_type" in filtered.columns:
            filtered = filtered[
                filtered["error_type"].astype(str) == chosen_type
            ]

        if chosen_class != "All" and "class" in filtered.columns:
            filtered = filtered[
                filtered["class"].astype(str) == chosen_class
            ]

        st.metric("Matching audit records", len(filtered))
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "Download filtered audit CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="t231_filtered_audit.csv",
            mime="text/csv",
        )

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

    st.subheader("Important limitations")
    st.markdown(
        """
        - This is a research prototype, not a certified safety system.
        - A predicted PPE violation is not definitive proof of a violation.
        - Missed detections are possible; test recall is 69.53%.
        - Performance can change with camera angle, lighting, occlusion,
          distance and deployment conditions.
        - Bounding boxes and confidence scores explain the model's
          output at a basic level; they are not causal explanations.
        - This dashboard does not currently implement Grad-CAM,
          SHAP or a validated explainability method.
        - False-positive and false-negative counts depend on the chosen
          confidence threshold and matching IoU.
        """
    )

st.markdown("---")
st.caption(
    "T231 research prototype | YOLO object detection and error auditing"
)
