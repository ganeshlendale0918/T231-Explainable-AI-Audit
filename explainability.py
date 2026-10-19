import numpy as np
from PIL import Image


def occlusion_sensitivity(
    image,
    model,
    target_class_id,
    target_box,
    original_score,
    conf=0.05,
    iou=0.45,
    grid=6,
    imgsz=640,
):
    """
    Model-agnostic occlusion sensitivity for one YOLO detection.

    Each grid cell is replaced with its local mean color.
    Importance measures the reduction in a matching detection's
    confidence. This is a perturbation-based explanation, not Grad-CAM.
    """
    from PIL import ImageDraw

    rgb = image.convert("RGB")
    original = np.asarray(rgb).copy()
    height, width = original.shape[:2]
    x1, y1, x2, y2 = map(float, target_box)

    heat = np.zeros((grid, grid), dtype=np.float32)

    for row in range(grid):
        for col in range(grid):
            xa = int(col * width / grid)
            xb = int((col + 1) * width / grid)
            ya = int(row * height / grid)
            yb = int((row + 1) * height / grid)

            masked = original.copy()
            patch = masked[ya:yb, xa:xb]

            if patch.size == 0:
                continue

            fill = np.mean(patch, axis=(0, 1)).astype(np.uint8)
            masked[ya:yb, xa:xb] = fill
            masked_image = Image.fromarray(masked)

            prediction = model.predict(
                source=masked_image,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
            )[0]

            best_score = 0.0
            boxes = prediction.boxes

            if boxes is not None:
                for box in boxes:
                    if int(box.cls.item()) != int(target_class_id):
                        continue

                    candidate = box.xyxy[0].cpu().numpy()
                    ax1 = max(x1, float(candidate[0]))
                    ay1 = max(y1, float(candidate[1]))
                    ax2 = min(x2, float(candidate[2]))
                    ay2 = min(y2, float(candidate[3]))

                    intersection = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                    area_a = max(0, x2 - x1) * max(0, y2 - y1)
                    area_b = max(0, candidate[2] - candidate[0]) * max(
                        0, candidate[3] - candidate[1]
                    )
                    union = area_a + area_b - intersection
                    overlap = intersection / union if union > 0 else 0

                    if overlap >= 0.30:
                        best_score = max(
                            best_score, float(box.conf.item())
                        )

            heat[row, col] = max(0.0, float(original_score) - best_score)

    # Expand the grid to image size and normalize for visualization.
    heat_image = Image.fromarray(
        np.uint8(np.clip(heat / max(float(heat.max()), 1e-8), 0, 1) * 255)
    ).resize((width, height), Image.Resampling.BILINEAR)

    heatmap = np.asarray(heat_image, dtype=np.float32) / 255.0

    # Create a simple red/yellow overlay without extra plotting libraries.
    colored = np.zeros_like(original, dtype=np.float32)
    colored[:, :, 0] = np.clip(heatmap * 2.0, 0, 1) * 255
    colored[:, :, 1] = np.clip((heatmap - 0.35) * 1.7, 0, 1) * 255

    alpha = (heatmap * 0.65)[:, :, None]
    overlay = original * (1 - alpha) + colored * alpha
    overlay = np.uint8(np.clip(overlay, 0, 255))

    return Image.fromarray(overlay), Image.fromarray(
        np.uint8(heatmap * 255)
    )
