import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import correlate
import matplotlib.pyplot as plt

def load_image_as_array(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)

def create_circular_mask(h: int, w: int) -> np.ndarray:
    center_y, center_x = h / 2.0 - 0.5, w / 2.0 - 0.5
    radius = min(h, w) / 2.0
    y, x = np.ogrid[:h, :w]
    mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(np.float64)
    return mask

def masked_ncc_single_channel(F: np.ndarray, H: np.ndarray, mask: np.ndarray) -> np.ndarray:
    N_mask = np.sum(mask)
    eps = 1e-10

    H_masked = H * mask
    H_rms = np.sqrt(np.sum(H_masked**2) / N_mask)
    H_norm = H_masked / (H_rms + eps)

    numerator = correlate(F, H_norm, mode="valid", method="fft")

    F_sq_sum_masked = correlate(F**2, mask, mode="valid", method="fft")
    F_rms_masked = np.sqrt(np.maximum(F_sq_sum_masked, 0) / N_mask)

    scores = numerator / (F_rms_masked + eps)
    return scores

def ncc_match_color_masked(F_rgb: np.ndarray, H_rgb: np.ndarray, mask: np.ndarray):
    """Computes Masked NCC for each channel independently, returning a dictionary of scores."""
    channel_names = ["R", "G", "B"]
    per_channel_scores = {}

    for i, name in enumerate(channel_names):
        F_ch = F_rgb[:, :, i]
        H_ch = H_rgb[:, :, i]
        print(f"Computing Masked NCC for channel {name}...")
        per_channel_scores[name] = masked_masked_single_channel_wrapper = masked_ncc_single_channel(F_ch, H_ch, mask)

    return per_channel_scores

def find_best_match(scores: np.ndarray):
    idx = np.unravel_index(np.argmax(scores), scores.shape)
    row, col = idx
    return row, col, scores[row, col]

def linear_interpolate(x, x0, x1, y0, y1):
    if x0 == x1:
        return float(y0)
    numerator = y0 * (x1 - x) + y1 * (x - x0)
    denominator = x1 - x0
    return numerator / denominator

def bilinear_resize_single_channel(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    image_float = image.astype(np.float32)
    M, N = image_float.shape

    row_scale = (M - 1) / (new_h - 1) if new_h > 1 else 0
    col_scale = (N - 1) / (new_w - 1) if new_w > 1 else 0

    resized = np.zeros((new_h, new_w), dtype=np.float32)

    for r in range(new_h):
        orig_r = r * row_scale
        r0 = int(np.floor(orig_r))
        r1 = r0 + 1 if r0 + 1 < M else M - 1

        for c in range(new_w):
            orig_c = c * col_scale
            c0 = int(np.floor(orig_c))
            c1 = c0 + 1 if c0 + 1 < N else N - 1

            I_00 = image_float[r0, c0]
            I_01 = image_float[r0, c1]
            I_10 = image_float[r1, c0]
            I_11 = image_float[r1, c1]

            val_top = linear_interpolate(orig_c, c0, c1, I_00, I_01)
            val_bottom = linear_interpolate(orig_c, c0, c1, I_10, I_11)
            final_intensity = linear_interpolate(orig_r, r0, r1, val_top, val_bottom)

            resized[r, c] = final_intensity

    return resized

def bilinear_resize_rgb(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    channels = []
    for c in range(image.shape[2]):
        resized_channel = bilinear_resize_single_channel(image[:, :, c], new_h, new_w)
        channels.append(resized_channel)
    return np.stack(channels, axis=2)

def draw_match_box(base_rgb: np.ndarray, top_left, box_h: int, box_w: int):
    row, col = top_left
    img = Image.fromarray(np.clip(base_rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([col, row, col + box_w, row + box_h], outline=(255, 0, 0), width=3)
    return np.array(img)

def save_combined_masked_report(scene_rgb, per_channel_scores, template_shape, out_path="masked_combined_results.jpg"):
    """Generates the dual-row report figure for Masked NCC: Top = Heatmaps, Bottom = Best Matches."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    channel_names = ["R", "G", "B"]
    
    th, tw = template_shape[:2]
    offset_r = th // 2
    offset_c = tw // 2

    for i, name in enumerate(channel_names):
        scores = per_channel_scores[name]
        row, col, score = find_best_match(scores)
        
        adjusted_top_left = (row + offset_r, col + offset_c)
        
        # --- Top Row: Heatmaps ---
        ax_heat = axes[0, i]
        im_heat = ax_heat.imshow(scores, cmap='hot')
        ax_heat.set_title(f"Channel {name} Heatmap")
        ax_heat.axis('off')
        fig.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.04)

        # --- Bottom Row: Best Matches with Bounding Box ---
        ax_match = axes[1, i]
        annotated_scene = draw_match_box(scene_rgb, adjusted_top_left, th, tw)
        ax_match.imshow(annotated_scene)
        ax_match.set_title(f"Channel {name} Match (score={score:.4f})")
        ax_match.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved masked combined report visualization to {out_path}")
    plt.close()

def main(scene_path: str, template_path: str):
    scene = load_image_as_array(scene_path)
    template = load_image_as_array(template_path)

    print(f"Scene shape:    {scene.shape}")
    print(f"Template shape: {template.shape}\n")

    # Create circular mask based on template dimensions
    th, tw = template.shape[:2]
    mask = create_circular_mask(th, tw)

    # Compute Masked NCC scores for R, G, and B independently
    per_channel_scores = ncc_match_color_masked(scene, template, mask)

    for name, scores in per_channel_scores.items():
        row, col, score = find_best_match(scores)
        print(f"Channel {name}: Best match at (row={row}, col={col}) with score={score:.4f}")

    # Generate and save the dual-row visualization report
    save_combined_masked_report(scene, per_channel_scores, template.shape, out_path=f"masked_combined_results_size{th}.jpg")

if __name__ == "__main__":
    scene_path = "parking.png"
    template_path = "templateNoPark.png"
    
    main(scene_path, template_path)
