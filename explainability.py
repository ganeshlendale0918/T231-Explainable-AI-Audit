"""
Explainability helpers for the T231 Audit Dashboard.
=====================================================
Includes:
  - **Grad-CAM** (gradient-weighted class activation mapping)
    Uses PyTorch hooks (register_forward_hook + register_full_backward_hook)
    on the last Conv2d layer of the YOLO backbone.
  - **Score-CAM** (score-weighted, gradient-free variant)
    Extracts activation maps via forward hook, masks images per channel,
    forward-passes each, and weights channels by detection confidence.
  - **Occlusion Sensitivity** (perturbation-based, kept for comparison)

All methods produce:
  overlay : PIL.Image  — heatmap blended on top of the original image
  heatmap : PIL.Image  — standalone heatmap (uint8, single channel)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


# --------------------------------------------------------------------------- #
#  Grad-CAM for YOLO
# --------------------------------------------------------------------------- #

class YOLOGradCAM:
    """
    Grad-CAM implementation for Ultralytics YOLO models.

    Hooks into the last convolutional layer of the backbone to capture
    activations and gradients, then produces a class-activation heatmap
    for the selected detection box.

    Implementation notes
    --------------------
    - ``register_forward_hook``  captures feature-map activations.
    - ``register_full_backward_hook`` captures gradient flowing back through
      the same layer.
    - Grad-CAM weights = global-average-pooled gradients (per channel).
    - The final CAM = ReLU(Σ_c weight_c * activation_c).
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    # -- hook helpers -------------------------------------------------------- #

    def _get_target_layer(self):
        """Return the last Conv2d in the YOLO backbone."""
        backbone = self.model.model.model          # nn.Sequential
        # Walk backwards to find the last module that contains a Conv2d
        for layer in reversed(list(backbone.children())):
            for module in reversed(list(layer.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    return module
        # Fallback: use the backbone output directly
        return backbone[-1]

    def _register_hooks(self):
        target = self._get_target_layer()
        self._hooks.append(
            target.register_forward_hook(self._fwd_hook)
        )
        self._hooks.append(
            target.register_full_backward_hook(self._bwd_hook)
        )

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -- public API ---------------------------------------------------------- #

    def generate(self, image: Image.Image, target_box, target_class_id,
                 imgsz=640):
        """
        Run Grad-CAM for *one* detection.

        Parameters
        ----------
        image : PIL.Image
        target_box : list[float]   – [x1, y1, x2, y2] in pixel coords
        target_class_id : int
        imgsz : int

        Returns
        -------
        overlay : PIL.Image   – heatmap blended on top of original
        heatmap : PIL.Image   – standalone heatmap (JET coloured)
        """
        rgb = image.convert("RGB")
        w_orig, h_orig = rgb.size

        # Prepare input tensor the same way YOLO does
        img_np = np.asarray(rgb)
        img_resized = cv2.resize(img_np, (imgsz, imgsz))
        tensor = (
            torch.from_numpy(img_resized)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )

        device = next(self.model.model.parameters()).device
        tensor = tensor.to(device)
        tensor.requires_grad_(True)

        # Forward
        self.model.model.eval()
        preds = self.model.model(tensor)          # raw predictions

        # Build a scalar target: sum of objectness × class-prob for matching
        # detections (we aggregate across candidate anchors that overlap the
        # target box).
        loss = self._build_target(preds, target_box, target_class_id,
                                   w_orig, h_orig, imgsz)

        # Back-prop
        self.model.model.zero_grad()
        loss.backward()

        # Compute Grad-CAM weights
        if self.gradients is None or self.activations is None:
            return self._fallback(rgb)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # GAP
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()

        # Handle edge case: cam might be 0-dim if there's a single element
        if cam.ndim == 0:
            return self._fallback(rgb)

        # Normalise
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size
        cam_resized = cv2.resize(cam, (w_orig, h_orig))
        cam_uint8 = np.uint8(cam_resized * 255)
        heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img_np, 0.55, heatmap_rgb, 0.45, 0)

        return Image.fromarray(overlay), Image.fromarray(heatmap_rgb)

    # -- internal ------------------------------------------------------------ #

    @staticmethod
    def _build_target(preds, target_box, cls_id, w_orig, h_orig, imgsz):
        """
        Derive a differentiable scalar from raw YOLO predictions that
        corresponds to the target detection.  Works for YOLOv8 output
        format (list of tensors).
        """
        x1, y1, x2, y2 = target_box
        # Normalised centre of target box
        # (not used for filtering right now but kept for potential future use)
        # cx = ((x1 + x2) / 2) / w_orig
        # cy = ((y1 + y2) / 2) / h_orig

        # preds can be a tuple/list; take the first detection head
        if isinstance(preds, (list, tuple)):
            pred = preds[0]
        else:
            pred = preds

        # pred shape: (B, num_outputs, num_anchors) for v8
        # num_outputs = 4 (box) + num_classes
        if pred.dim() == 3:
            pred = pred.squeeze(0)          # (num_outputs, num_anchors)
            num_cls = pred.shape[0] - 4
            if cls_id < num_cls:
                cls_scores = pred[4 + cls_id]   # (num_anchors,)
            else:
                cls_scores = pred[4:].max(dim=0).values

            # Use top-K highest class scores as target
            num_anchors = cls_scores.shape[0]
            topk = min(10, num_anchors)
            vals, _ = cls_scores.topk(topk)
            return vals.sum()

        # Fallback for unexpected shapes
        return pred.sum() * 0.0 + 1.0

    @staticmethod
    def _fallback(rgb):
        """Return an empty heatmap when hooks failed."""
        w, h = rgb.size
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        return Image.fromarray(np.asarray(rgb)), Image.fromarray(blank)


# --------------------------------------------------------------------------- #
#  Score-CAM for YOLO (gradient-free)
# --------------------------------------------------------------------------- #

def score_cam(image: Image.Image, model, target_class_id, target_box,
              imgsz=640, conf=0.05, iou=0.45, top_k_channels=16):
    """
    Score-CAM: no gradients required.

    Algorithm
    ---------
    1. Register a forward hook on the last Conv2d of the backbone.
    2. Run a normal prediction to capture activation maps.
    3. For each of the top-K activation channels (by L2 norm):
       a. Normalise the channel to [0, 1].
       b. Up-sample to image size and use as a soft mask.
       c. Forward-pass the masked image through YOLO.
       d. Record detection confidence for the target box/class.
    4. Weight channels by confidence scores and combine → heatmap.
    """
    rgb = image.convert("RGB")
    img_np = np.asarray(rgb)
    w_orig, h_orig = rgb.size

    # ---- step 1: get activations ------------------------------------------ #
    activations = {}

    def hook_fn(module, inp, out):
        activations["feat"] = out.detach()

    # Find last Conv2d
    backbone = model.model.model
    target_layer = None
    for layer in reversed(list(backbone.children())):
        for module in reversed(list(layer.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        if target_layer:
            break

    if target_layer is None:
        # Can't find layer – return blank
        blank = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        return Image.fromarray(img_np), Image.fromarray(blank)

    handle = target_layer.register_forward_hook(hook_fn)

    # Run a normal prediction to capture activations
    model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    handle.remove()

    if "feat" not in activations:
        blank = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
        return Image.fromarray(img_np), Image.fromarray(blank)

    feat = activations["feat"].squeeze(0)    # (C, H', W')
    C = feat.shape[0]

    # Limit channels for speed
    if C > top_k_channels:
        norms = feat.flatten(1).norm(dim=1)
        _, top_idx = norms.topk(top_k_channels)
        feat = feat[top_idx]
        C = top_k_channels

    # ---- step 2-3: mask + forward ----------------------------------------- #
    feat_np = feat.cpu().numpy()
    scores = np.zeros(C, dtype=np.float32)

    for ch_i in range(C):
        # Normalise activation channel to [0, 1]
        activation_map = feat_np[ch_i]
        a_min, a_max = activation_map.min(), activation_map.max()
        if a_max - a_min > 1e-8:
            activation_map = (activation_map - a_min) / (a_max - a_min)
        else:
            activation_map = np.zeros_like(activation_map)

        # Resize to image dimensions and apply as mask
        mask = cv2.resize(activation_map, (w_orig, h_orig))
        masked = (img_np * mask[:, :, None]).astype(np.uint8)
        masked_pil = Image.fromarray(masked)

        result = model.predict(
            source=masked_pil, conf=conf, iou=iou, imgsz=imgsz, verbose=False
        )[0]

        # Find best matching detection
        best = 0.0
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls.item()) != int(target_class_id):
                    continue
                cand = box.xyxy[0].cpu().numpy().tolist()
                iou_val = _box_iou(target_box, cand)
                if iou_val >= 0.30:
                    best = max(best, float(box.conf.item()))
        scores[ch_i] = best

    # ---- step 4: weighted combination ------------------------------------- #
    cam = np.zeros((feat_np.shape[1], feat_np.shape[2]), dtype=np.float32)
    for ch_i in range(C):
        cam += scores[ch_i] * feat_np[ch_i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    cam_resized = cv2.resize(cam, (w_orig, h_orig))
    cam_uint8 = np.uint8(cam_resized * 255)
    heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img_np, 0.55, heatmap_rgb, 0.45, 0)

    return Image.fromarray(overlay), Image.fromarray(heatmap_rgb)


def _box_iou(box_a, box_b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Occlusion sensitivity (original, kept for comparison)
# --------------------------------------------------------------------------- #

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
    """
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
                source=masked_image, conf=conf, iou=iou,
                imgsz=imgsz, verbose=False,
            )[0]

            best_score = 0.0
            boxes = prediction.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls.item()) != int(target_class_id):
                        continue
                    candidate = box.xyxy[0].cpu().numpy()
                    overlap = _box_iou(
                        [x1, y1, x2, y2],
                        [float(candidate[0]), float(candidate[1]),
                         float(candidate[2]), float(candidate[3])]
                    )
                    if overlap >= 0.30:
                        best_score = max(best_score, float(box.conf.item()))

            heat[row, col] = max(0.0, float(original_score) - best_score)

    # Normalise and resize to image dimensions
    heat_max = float(heat.max())
    if heat_max > 1e-8:
        heat_norm = heat / heat_max
    else:
        heat_norm = heat

    heat_resized = cv2.resize(
        np.uint8(np.clip(heat_norm, 0, 1) * 255),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_float = heat_resized.astype(np.float32) / 255.0
    heatmap_colored = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    alpha = (heatmap_float * 0.55)[:, :, None]
    overlay = original.astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha
    overlay = np.uint8(np.clip(overlay, 0, 255))

    return Image.fromarray(overlay), Image.fromarray(heatmap_rgb)
