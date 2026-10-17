import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.signal import correlate

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

    numerator = correlate(F, H_norm, mode="valid", method="fft")

    F_sq_sum = correlate(F ** 2, np.ones_like(H), mode="valid", method="fft")
    F_rms = np.sqrt(np.maximum(F_sq_sum, 0) / N)

    scores = numerator / (F_rms + eps)
    return scores

def find_best_match(scores: np.ndarray):
    idx = np.unravel_index(np.argmax(scores), scores.shape)
    row, col = idx
    return row, col, scores[row, col]

def gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

def convolve_2d(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    radius = len(kernel_1d) // 2
    H, W = img.shape

    padded = np.pad(img, radius, mode="edge")

    temp = np.zeros((H, W))
    for j in range(W):
        region = padded[radius:radius + H, j:j + len(kernel_1d)]
        temp[:, j] = region @ kernel_1d

    padded2 = np.pad(temp, radius, mode="edge")
    out = np.zeros((H, W))
    for i in range(H):
        region = padded2[i:i + len(kernel_1d), radius:radius + W]
        out[i, :] = kernel_1d @ region

    return out


def downsample_by_factor(img: np.ndarray, factor: int) -> np.ndarray:
    sigma = factor / 2.0
    radius = int(3 * sigma)
    kernel = gaussian_kernel_1d(sigma, radius)

    if img.ndim == 2:
        blurred = convolve_2d(img, kernel)
        return blurred[::factor, ::factor]

    channels = []
    for c in range(img.shape[2]):
        blurred = convolve_2d(img[:, :, c], kernel)
        channels.append(blurred[::factor, ::factor])
    return np.stack(channels, axis=2)

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
    draw.rectangle([col, row, col + box_w, row + box_h], outline=(255, 0, 0), width=2)
    img.save(out_path)
    print(f"Saved annotated result to {out_path}")
    
    return np.array(img)


def main(scene_path: str, template_path: str):
    scene = load_image_as_array(scene_path)
    template = load_image_as_array(template_path)

    print(f"Scene shape:    {scene.shape}")
    print(f"Template shape: {template.shape}")

    scene_small = downsample_by_factor(scene, 5)
    print(f"Downsampled scene shape: {scene_small.shape}")

    template_sizes = [41, 51, 61]
    channel_names = ["R", "G", "B"]

    for size in template_sizes:
        template_resized = bilinear_resize_rgb(template, size, size)
        print(f"\n=== Template size {size}x{size} ===")

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(f"Template {size}x{size} — Top: NCC Scores | Bottom: Best Matches", fontsize=14)

        for i, name in enumerate(channel_names):
            F_ch = scene_small[:, :, i]
            H_ch = template_resized[:, :, i]

            scores = ncc_single_channel(F_ch, H_ch)
            row, col, score = find_best_match(scores)
            print(f"  Channel {name}: best match at (row={row}, col={col}), score={score:.4f}")

            im = axes[0, i].imshow(scores, cmap="hot")
            axes[0, i].set_title(f"Channel {name} Heatmap")
            axes[0, i].axis("off")
            fig.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

            out_path = f"match_size{size}_{name}.png"
            
            annotated_arr = draw_match_box(
                scene_small, (row, col), box_h=size, box_w=size, out_path=out_path
            )
            
            axes[1, i].imshow(annotated_arr)
            axes[1, i].set_title(f"Channel {name} Match (score={score:.4f})")
            axes[1, i].axis("off")

        plt.tight_layout()
        
        plt.savefig(f"combined_results_size{size}.png")
        
        plt.show()

if __name__ == "__main__":
    main("parking.png", "templateNoPark.png")
