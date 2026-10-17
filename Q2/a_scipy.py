import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import correlate
import matplotlib.pyplot as plt
import os

def load_image_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)

def rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(x ** 2))

def ncc_single_channel(F: np.ndarray, H: np.ndarray) -> np.ndarray:
    Hgt_H, Wid_H = H.shape
    N = Hgt_H * Wid_H
    eps = 1e-10

    H_norm = H / (rms(H) + eps)

    # Numerator: sum(Fpatch * H_norm) for every window position, via correlation
    numerator = correlate(F, H_norm, mode="valid", method="fft")

    # Sliding sum of squares of F, via correlation with a ones kernel
    F_sq_sum = correlate(F ** 2, np.ones_like(H), mode="valid", method="fft")
    F_rms = np.sqrt(np.maximum(F_sq_sum, 0) / N)  # clip tiny negatives from FFT round-off

    scores = numerator / (F_rms + eps)
    return scores

def ncc_match_color(F_rgb: np.ndarray, H_rgb: np.ndarray):
    """Computes NCC for each channel independently, returning a dictionary of scores."""
    channel_names = ["R", "G", "B"]
    per_channel_scores = {}

    for i, name in enumerate(channel_names):
        F_ch = F_rgb[:, :, i]
        H_ch = H_rgb[:, :, i]
        print(f"Computing NCC for channel {name}...")
        per_channel_scores[name] = ncc_single_channel(F_ch, H_ch)

    return per_channel_scores

def find_best_match(scores: np.ndarray):
    idx = np.unravel_index(np.argmax(scores), scores.shape)
    row, col = idx
    return row, col, scores[row, col]

def draw_match_box(scene_rgb: np.ndarray, top_left, template_shape):
    """Draws bounding box on a copy of the scene and returns the annotated RGB array."""
    h, w = template_shape[:2]
    row, col = top_left

    img = Image.fromarray(np.clip(scene_rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([col, row, col + w, row + h], outline=(255, 0, 0), width=3)
    return np.array(img)

def save_combined_report(scene_rgb, per_channel_scores, template_shape, out_path="combined_results.jpg"):
    """Generates the dual-row report figure: Top = Heatmaps, Bottom = Best Matches."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    channel_names = ["R", "G", "B"]
    
    # Calculate offset shift due to 'valid' convolution mode
    th, tw = template_shape[:2]
    offset_r = th // 2
    offset_c = tw // 2

    for i, name in enumerate(channel_names):
        scores = per_channel_scores[name]
        row, col, score = find_best_match(scores)
        
        # Adjust top-left coordinate to align with the original scene dimensions
        adjusted_top_left = (row + offset_r, col + offset_c)
        
        # --- Top Row: Heatmaps ---
        ax_heat = axes[0, i]
        im_heat = ax_heat.imshow(scores, cmap='hot')
        ax_heat.set_title(f"Channel {name} Heatmap")
        ax_heat.axis('off')
        fig.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)

        # --- Bottom Row: Best Matches with Bounding Box ---
        ax_match = axes[1, i]
        annotated_scene = draw_match_box(scene_rgb, adjusted_top_left, template_shape)
        ax_match.imshow(annotated_scene)
        ax_match.set_title(f"Channel {name} Match (score={score:.4f})")
        ax_match.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined report visualization to {out_path}")
    plt.close()

def main(scene_path: str, template_path: str):
    scene = load_image_as_array(scene_path)
    template = load_image_as_array(template_path)

    print(f"Scene shape:    {scene.shape}")
    print(f"Template shape: {template.shape}\n")

    # Get independent scores for R, G, and B
    per_channel_scores = ncc_match_color(scene, template)

    # Find best matches and print details
    for name, scores in per_channel_scores.items():
        row, col, score = find_best_match(scores)
        print(f"Channel {name}: Best match at (row={row}, col={col}) with score={score:.4f}")

    # Generate and save the professional dual-row visualization
    save_combined_report(scene, per_channel_scores, template.shape, out_path="combined_results_template.jpg")

if __name__ == "__main__":
    scene_path = "parking.png"
    template_path = "templateNoPark.png"
    
    main(scene_path, template_path)
