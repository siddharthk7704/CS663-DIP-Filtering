import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def linear_interpolate_array(x, x0, x1, y0, y1):
    safe_denom = np.where(x0 == x1, 1.0, x1 - x0)
    res = (y0 * (x1 - x) + y1 * (x - x0)) / safe_denom
    return np.where(x0 == x1, y0, res)

def get_separable_gaussian(sigma=1.0, size = 5.0):
    center = size // 2
    x = np.arange(-center, center + 1)
    g_1d = np.exp(-(x**2) / (2 * sigma**2))
    g_1d /= np.sum(g_1d)
    return g_1d.reshape(-1, 1), g_1d.reshape(1, -1)

def loop_convolution_separable(image, U, V):
    H, W = image.shape
    ku, kv = U.shape[0], V.shape[1]
    
    pad_vert = np.pad(image, ((ku//2, ku//2), (0, 0)), mode='edge')
    temp = np.zeros_like(image)
    for i in range(H):
        for j in range(W):
            temp[i, j] = np.sum(pad_vert[i:i+ku, j:j+1] * U)
            
    pad_horiz = np.pad(temp, ((0, 0), (kv//2, kv//2)), mode='edge')
    out = np.zeros_like(image)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(pad_horiz[i:i+1, j:j+kv] * V)
            
    return out

def to_gray(image):
    image = np.asarray(image).astype(np.float64)
    if image.ndim == 2:
        return image
    return (
        0.299 * image[..., 0]
        + 0.587 * image[..., 1]
        + 0.114 * image[..., 2]
    )

def nms_interpolated(score, Ix, Iy):

    M, N = score.shape
    out = np.zeros((M, N), dtype=np.float32)
    
    v_norm = np.sqrt(Ix**2 + Iy**2) + 1e-8
    
    u_x = Ix / v_norm
    u_y = Iy / v_norm
    
    y, x = np.meshgrid(np.arange(1, M-1), np.arange(1, N-1), indexing='ij')
    
    u_y_in = u_y[1:-1, 1:-1]
    u_x_in = u_x[1:-1, 1:-1]
    score_in = score[1:-1, 1:-1]
    
    def get_interp(y_c, x_c):
        y0 = np.clip(np.floor(y_c).astype(int), 0, M-2)
        x0 = np.clip(np.floor(x_c).astype(int), 0, N-2)
        y1, x1 = y0 + 1, x0 + 1
        
        val_top = linear_interpolate_array(x_c, x0, x1, score[y0, x0], score[y0, x1])
        val_bottom = linear_interpolate_array(x_c, x0, x1, score[y1, x0], score[y1, x1])
        return linear_interpolate_array(y_c, y0, y1, val_top, val_bottom)

    score_plus_u = get_interp(y + u_y_in, x + u_x_in)
    score_minus_u = get_interp(y - u_y_in, x - u_x_in)
    
    keep = (score_in > score_plus_u) & (score_in > score_minus_u)
    out[1:-1, 1:-1][keep] = score_in[keep]
    
    return out

def hysteresis_thresholding(nms, low_threshold, high_threshold):
    strong = nms >= high_threshold
    weak = (nms >= low_threshold) & (nms < high_threshold)
    
    edges = strong.copy()
    
    while True:
        dilated = edges.copy()
        dilated[:-1, :] |= edges[1:, :]
        dilated[1:, :] |= edges[:-1, :]
        dilated[:, :-1] |= edges[:, 1:]
        dilated[:, 1:] |= edges[:, :-1]
        dilated[:-1, :-1] |= edges[1:, 1:]
        dilated[1:, 1:] |= edges[:-1, :-1]
        dilated[:-1, 1:] |= edges[1:, :-1]
        dilated[1:, :-1] |= edges[:-1, 1:]
        
        new_edges = (dilated & weak) | edges
        
        if np.array_equal(new_edges, edges):
            break
        edges = new_edges

    return (edges * 255).astype(np.uint8)

def canny_edge_detection(image, sigma=1.4, low_pct=20, high_pct=45):
    gray = to_gray(image)
    
    U_g, V_g = get_separable_gaussian(sigma=sigma, size = 5)
    smoothed = loop_convolution_separable(gray, U_g, V_g)

    U_sy, V_sx = np.array([[1], [2], [1]], dtype=np.float64), np.array([[-1, 0, 1]], dtype=np.float64)
    U_sy_deriv, V_sx_smooth = np.array([[-1], [0], [1]], dtype=np.float64), np.array([[1, 2, 1]], dtype=np.float64)
    
    gx = loop_convolution_separable(smoothed, U_sy, V_sx)
    gy = loop_convolution_separable(smoothed, U_sy_deriv, V_sx_smooth)

    magnitude = np.hypot(gx, gy)

    nms = nms_interpolated(magnitude, gx, gy)
    
    max_nms = nms.max()
    high_threshold = (high_pct / 100.0) * max_nms
    low_threshold = (low_pct / 100.0) * max_nms
    high_threshold = max(high_threshold, low_threshold + 1e-6)

    edges = hysteresis_thresholding(nms, low_threshold, high_threshold)
    return edges

def overlay_edges_black(image, edges):
    overlay = np.array(image, dtype=np.uint8, copy=True)
    edge_mask = edges > 0
    overlay[edge_mask] = [0, 0, 0]
    return overlay

def show_edge_pipeline(image_path, sigma=1.4, low_pct=20, high_pct=45, save_path=None):
    image = np.asarray(Image.open(image_path).convert("RGB"))

    edges = canny_edge_detection(image, sigma=sigma, low_pct=low_pct, high_pct=high_pct)
    overlay = overlay_edges_black(image, edges)

    if save_path:
        Image.fromarray(overlay).save(f"{save_path}_overlay.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Detected Edges (Binary)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Edges Overlaid on Input (Black)")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return image, edges, overlay

if __name__ == "__main__":
    image_paths = {
        "Butterfly": "butterfly.png",
        "Paithani": "paithaniEdge.png",
        "Rangoli": "rangoli.png",
    }

    print("\nProcessing Butterfly...")
    show_edge_pipeline(image_paths["Butterfly"], sigma=1.0, low_pct=15, high_pct=30, save_path="edges_butterfly.png")

    print("\nProcessing Paithani...")
    show_edge_pipeline(image_paths["Paithani"], sigma=1.6, low_pct=20, high_pct=45, save_path="edge_paithani.png")

    print("\nProcessing Rangoli...")
    show_edge_pipeline(image_paths["Rangoli"], sigma=1.0, low_pct=10, high_pct=25, save_path="edges_rangoli.png")
