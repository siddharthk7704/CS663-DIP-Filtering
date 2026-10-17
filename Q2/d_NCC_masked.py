import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.signal import correlate


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


def draw_match_box(base_rgb: np.ndarray, top_left, box_h: int, box_w: int, out_path: str):
    row, col = top_left

    img = Image.fromarray(np.clip(base_rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    draw.rectangle([col, row, col + box_w, row + box_h], outline=(255, 0, 0), width=3)
    img.save(out_path)
    print(f"Saved annotated result to {out_path}")
    
    return np.array(img)

def main(scene_path: str, template_path: str):
    scene = load_image_as_array(scene_path)
    template = load_image_as_array(template_path)

    print(f"Scene shape:    {scene.shape}")
    print(f"Template shape: {template.shape}")

    template_sizes = [201, 251, 301]
    channel_names = ["R", "G", "B"]

    for size in template_sizes:
        print(f"\n=== Resizing template to {size}x{size} ===")
        template_resized = bilinear_resize_rgb(template, size, size)

        mask = create_circular_mask(size, size)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(f"Masked NCC — Template {size}x{size} | Top: Scores | Bottom: Best Matches", fontsize=14)

        for i, name in enumerate(channel_names):
            print(f"Computing Masked NCC for Channel {name}...")
            F_ch = scene[:, :, i]
            H_ch = template_resized[:, :, i]

            scores = masked_ncc_single_channel(F_ch, H_ch, mask)
            row, col, score = find_best_match(scores)
            print(f"  Channel {name}: best match at (row={row}, col={col}), score={score:.4f}")

            im = axes[0, i].imshow(scores, cmap="hot")
            axes[0, i].set_title(f"Channel {name} Heatmap")
            axes[0, i].axis("off")
            fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

            out_path = f"match_size{size}_{name}.png"
            annotated_arr = draw_match_box(
                scene, (row, col), box_h=size, box_w=size, out_path=out_path
            )
            
            axes[1, i].imshow(annotated_arr)
            axes[1, i].set_title(f"Channel {name} Match (score={score:.4f})")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(f"masked_combined_results_size{size}.png")
        plt.show()


if __name__ == "__main__":
    main("parking.png", "templateNoPark.png")
