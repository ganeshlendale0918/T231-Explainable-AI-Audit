
# T231 — Explainable AI Audit Dashboard

An interactive research prototype for auditing a YOLOv8n object-detection model on a construction-site worker-safety dataset.

## Features

- Image-based object detection
- Adjustable confidence and NMS IoU thresholds
- Detection results with CSV export
- Model evaluation metrics
- Class-wise false-positive and false-negative analysis
- Error-audit image galleries
- Experimental occlusion-sensitivity heatmaps
- Downloadable heatmap visualizations

## Test Results

| Metric | Result |
|---|---:|
| Precision | 85.76% |
| Recall | 69.53% |
| mAP@0.50 | 74.94% |
| mAP@0.50:0.95 | 45.20% |
| Test images | 82 |
| Ground-truth objects | 760 |
| Parameters | 3,007,598 |
| GFLOPs | 8.1 |

Results apply to the evaluated checkpoint and test configuration; they do not guarantee real-world performance.

## Technology Stack

- Python
- Streamlit
- Ultralytics YOLOv8n
- OpenCV
- Pandas
- NumPy
- Matplotlib

## Run Locally

Create and activate a virtual environment in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

## Explainability

The dashboard includes experimental perturbation-based occlusion sensitivity. It measures how model predictions change when image regions are masked.

This visualization is not Grad-CAM or SHAP and is not a validated causal explanation.

## Limitations

- The model may miss safety-related objects.
- Performance can vary with lighting, camera angle, distance, and occlusion.
- A predicted PPE violation is not definitive proof of a violation.
- The dashboard is a research prototype, not a certified safety system.

## Future Work

- Improve recall and analyze class-specific errors.
- Evaluate additional detection models.
- Validate explainability methods more rigorously.
- Test under varied deployment conditions.
