# FaceTag Crop (YOLO) — ComfyUI custom node

A ComfyUI custom node that uses an Ultralytics YOLO model to find a specific
bbox in an image (or a batch of video frames) and crops the image centered on
that detection, with temporal smoothing, normalization, and graceful fallbacks.

It outputs the cropped **image**, the aligned **mask**, and a **preview** with
the mask drawn on top.

## Install

Clone into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/adbrasi/facetaggg.git
```

Requires `ultralytics`, `opencv-python`, `numpy`, and `torch` (torch ships with
ComfyUI). Install ultralytics/opencv if you don't have them:

```bash
pip install ultralytics opencv-python
```

Put your YOLO models under `ComfyUI/models/ultralytics` (subfolders like
`bbox/` are searched recursively). The node lists them in a dropdown.

## Node: FaceTag Crop (YOLO)

Category: `image/facetag`

### Inputs

| Input | Description |
| --- | --- |
| `image` | A single image or a batch of frames (video). |
| `model_name` | YOLO model from `models/ultralytics` (recursive). |
| `confidence` | Detection confidence threshold. |
| `padding` | Crop size as a multiple of the detected box. |
| `output_width` / `output_height` | Output resolution. |
| `smoothing` | Temporal smoothing (zero-phase EMA) across the batch. |
| `zoom_mode` | `fixed` (one zoom for the whole batch) or `smooth`. |
| `zoom_percentile` | Percentile used to pick the fixed zoom level. |
| `mask_mode` | `follow_detection` (steady camera, mask tracks the box), `locked_center` (subject nailed to center, crop auto-zooms to stay inside the frame), or `full_frame` (NO crop — keep the original scene and just output the face mask; `padding` expands the mask box, width/height/zoom are ignored). |
| `max_gap` | Max missing-frame gap bridged by interpolation. |
| `mask_feather` | Gaussian blur radius on the mask edges (px). |
| `detector_imgsz` | Inference image size for the detector. |

### Outputs

- **image** — crop centered on the detection, at the chosen output size.
- **mask** — the detection as a mask, aligned to the crop.
- **preview** — the crop with the mask drawn as a translucent green overlay.

## Robustness

- Missing detections in some frames are interpolated; gaps larger than
  `max_gap` hold the nearest detection instead of sweeping across the frame.
- If **no** detection is found in the whole batch, it falls back to a
  full-frame crop and mask (with a console warning) instead of erroring.
- The YOLO model is cached between runs; uses CUDA + FP16 when available.
