import numpy as np
import matplotlib.pyplot as plt
import os

def linear_interpolate(x, x0, x1, y0, y1):
    if x0 == x1:
        return float(y0)
    numerator = y0 * (x1 - x) + y1 * (x - x0)
    denominator = x1 - x0
    return numerator / denominator

def linear_interpolate_array(x, x0, x1, y0, y1):
    safe_denom = np.where(x0 == x1, 1.0, x1 - x0)
    res = (y0 * (x1 - x) + y1 * (x - x0)) / safe_denom
    return np.where(x0 == x1, y0, res)

def get_separable_gaussian(size=5, sigma=1.0):
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

def compute_structure_tensor(gray_img, sigma=1.0):
    Ux, Vx = np.array([[1], [2], [1]], dtype=np.float32), np.array([[-1, 0, 1]], dtype=np.float32)
    Uy, Vy = np.array([[1], [0], [-1]], dtype=np.float32), np.array([[1, 2, 1]], dtype=np.float32)
    
    Ix = loop_convolution_separable(gray_img, Ux, Vx)
    Iy = loop_convolution_separable(gray_img, Uy, Vy)
    
    U_g, V_g = get_separable_gaussian(size=5, sigma=sigma)
    Sxx = loop_convolution_separable(Ix**2, U_g, V_g)
    Syy = loop_convolution_separable(Iy**2, U_g, V_g)
    Sxy = loop_convolution_separable(Ix*Iy, U_g, V_g)
    
    return Ix, Iy, Sxx, Syy, Sxy

def compute_eigenvalues(Sxx, Syy, Sxy):
    trace = Sxx + Syy
    delta = np.sqrt(np.maximum((Sxx - Syy)**2 + 4 * Sxy**2, 0))
    lambda_1 = (trace + delta) / 2.0
    lambda_2 = (trace - delta) / 2.0
    return lambda_1, lambda_2

def nms_2d_corners_fast(measure):
    pad = np.pad(measure, 1, mode='constant', constant_values=-np.inf)
    neighbors = np.stack([
        pad[:-2, :-2], pad[:-2, 1:-1], pad[:-2, 2:],
        pad[1:-1, :-2],                pad[1:-1, 2:],
        pad[2:, :-2],  pad[2:, 1:-1],  pad[2:, 2:]
    ])
    is_max = (measure > np.max(neighbors, axis=0)) & (measure > 0)
    return np.where(is_max, measure, 0)

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

def morphological_dilation(binary_map, radius=2):
    if radius == 0: return binary_map
    out = binary_map.copy()
    H, W = binary_map.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y_s = slice(max(0, dy), min(H, H + dy))
            x_s = slice(max(0, dx), min(W, W + dx))
            src_y_s = slice(max(0, -dy), min(H, H - dy))
            src_x_s = slice(max(0, -dx), min(W, W - dx))
            out[y_s, x_s] |= binary_map[src_y_s, src_x_s]
    return out

def run_harris(Ix, Iy, Sxx, Syy, Sxy, k_corner, c_thresh_frac, k_edge, e_thresh_frac):
    harris_c = (Sxx * Syy - Sxy**2) - k_corner * (Sxx + Syy)**2
    c_thresh_val = np.max(harris_c) * c_thresh_frac
    nms_corner = nms_2d_corners_fast(np.maximum(harris_c, 0))
    
    harris_e = (Sxx * Syy - Sxy**2) - k_edge * (Sxx + Syy)**2
    e_thresh_val = np.max(-harris_e) * e_thresh_frac
    nms_edge = nms_interpolated(np.maximum(-harris_e, 0), Ix, Iy)
    
    return {
        'C': harris_c,
        'nms_corner': nms_corner,
        'bin_corner': nms_corner > c_thresh_val,
        'nms_edge': nms_edge,
        'bin_edge': nms_edge > e_thresh_val
    }

def run_shi_tomasi(Sxx, Syy, Sxy, c_thresh_frac):
    lambda_1, lambda_2 = compute_eigenvalues(Sxx, Syy, Sxy)
    c_thresh_val = np.max(lambda_2) * c_thresh_frac
    nms_corner = nms_2d_corners_fast(lambda_2)
    
    return {
        'l1': lambda_1, 'l2': lambda_2, 
        'nms_corner': nms_corner, 
        'bin_corner': nms_corner > c_thresh_val
    }

if __name__ == "__main__":
    image_path = "nandadevi.png"
    
    HARRIS_K_CORNER = 0.08
    HARRIS_C_THRESH = 0.15
    HARRIS_K_EDGE = 0.08
    HARRIS_E_THRESH = 0.015
    SHI_C_THRESH = 0.25
    DOT_RADIUS = 2
    
    try:
        color_img = plt.imread(image_path)
        if np.max(color_img) <= 1.0: color_img = (color_img * 255).astype(np.float32)
        if color_img.ndim == 2: color_img = np.stack((color_img,)*3, axis=-1)
        elif color_img.shape[2] == 4: color_img = color_img[..., :3]
        gray_img = np.dot(color_img, [0.299, 0.587, 0.114])
            
    except FileNotFoundError:
        print(f"Error: Could not find {image_path}")
        exit()

    print(f"Processing full resolution image ({gray_img.shape[0]}x{gray_img.shape[1]})...")
    
    print("Computing Structure Tensors...")
    Ix, Iy, Sxx, Syy, Sxy = compute_structure_tensor(gray_img, sigma=1.0)
    
    print("Running Detectors...")
    harris = run_harris(Ix, Iy, Sxx, Syy, Sxy, HARRIS_K_CORNER, HARRIS_C_THRESH, HARRIS_K_EDGE, HARRIS_E_THRESH)
    shi = run_shi_tomasi(Sxx, Syy, Sxy, SHI_C_THRESH)
    
    print(f"Applying manual morphological dilation (Radius={DOT_RADIUS}) for overlays...")
    over_har_corn, over_shi_corn, over_har_edge = color_img.copy(), color_img.copy(), color_img.copy()
    
    over_har_corn[morphological_dilation(harris['bin_corner'], radius=DOT_RADIUS)] = [255, 0, 0]
    over_shi_corn[morphological_dilation(shi['bin_corner'], radius=DOT_RADIUS)] = [255, 0, 0] 
    over_har_edge[morphological_dilation(harris['bin_edge'], radius=1)] = [0, 255, 0] 
    
    displays = [
        (color_img.astype(np.uint8), "1 Input Image Color", None),
        (shi['l1'], "2 Largest Eval lambda1", 'viridis'),
        (shi['l2'], "3 Shi Tomasi Eval lambda2", 'viridis'),
        (harris['C'], "4 Harris Measure C", 'coolwarm'),
        (harris['nms_corner'], "5a NMS Harris Corners", 'hot'),
        (shi['nms_corner'], "5b NMS Shi Tomasi Corners", 'hot'),
        (harris['bin_corner'], "6a Binary Harris Corners", 'gray'),
        (shi['bin_corner'], "6b Binary Shi Tomasi Corners", 'gray'),
        (over_har_corn.astype(np.uint8), "7a Overlaid Harris Corners Color", None),
        (over_shi_corn.astype(np.uint8), "7b Overlaid Shi Tomasi Corners Color", None),
        (harris['nms_edge'], "8 NMS Harris Edges", 'hot'),
        (harris['bin_edge'], "9 Binary Harris Edges", 'gray'),
        (over_har_edge.astype(np.uint8), "10 Overlaid Harris Edges Color", None)
    ]

    out_dir = "corner_edge_outputs"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving outputs to '{out_dir}/'...")

    for data, title, cmap in displays:
        safe_name = "".join(c for c in title if c.isalnum() or c == ' ').strip()
        filename = f"{out_dir}/{safe_name.replace(' ', '_')}.png"
        vmin, vmax = (0, 1) if 'Binary' in title else (None, None)
        plt.imsave(filename, data, cmap=cmap, vmin=vmin, vmax=vmax)
        
    print("Execution complete.")
