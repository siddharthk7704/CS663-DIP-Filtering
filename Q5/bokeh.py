import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import os

def get_manual_mask(img):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_title("Click foreground object outline. Press ENTER when done.")
    
    if fig.canvas.toolbar:
        fig.canvas.toolbar.pan()
        fig.canvas.toolbar.zoom()
        
    points = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    
    H, W = img.shape[:2]
    y, x = np.mgrid[:H, :W]
    coord = np.stack((x.flatten(), y.flatten()), axis=-1)
    
    mask = Path(points).contains_points(coord).reshape((H, W))
    return mask.astype(np.float32)

def create_manual_bokeh_filter(D, sigma=1.0):
    R = D // 2
    y, x = np.ogrid[-R:R+1, -R:R+1]
    disc = ((x**2 + y**2) <= R**2).astype(np.float32)
    
    k_size = 5
    k_half = k_size // 2
    gy, gx = np.ogrid[-k_half:k_half+1, -k_half:k_half+1]
    
    gauss_kernel = np.exp(-(gx**2 + gy**2) / (2 * sigma**2), dtype=np.float32)
    gauss_kernel /= gauss_kernel.sum()
                          
    smoothed_disc = np.zeros_like(disc)
    pad_disc = np.pad(disc, k_half, mode='constant') 
    
    for i in range(D):
        for j in range(D):
            smoothed_disc[i, j] = np.sum(pad_disc[i:i+k_size, j:j+k_size] * gauss_kernel)
            
    smoothed_disc /= smoothed_disc.sum()
    return smoothed_disc

def manual_distance_transform(binary_bg):
    H, W = binary_bg.shape
    dist = np.full((H, W), np.inf)
    dist[binary_bg == 0] = 0 
    
    for i in range(H):
        for j in range(W):
            if i > 0: dist[i, j] = min(dist[i, j], dist[i-1, j] + 1)
            if j > 0: dist[i, j] = min(dist[i, j], dist[i, j-1] + 1)
            
    for i in range(H-1, -1, -1):
        for j in range(W-1, -1, -1):
            if i < H-1: dist[i, j] = min(dist[i, j], dist[i+1, j] + 1)
            if j < W-1: dist[i, j] = min(dist[i, j], dist[i, j+1] + 1)
            
    return dist

def apply_manual_bokeh(img, fg_mask, D):
    H, W, C = img.shape
    R = D // 2
    h_filter = create_manual_bokeh_filter(D)
    
    bg_mask = 1.0 - fg_mask 
    
    pad_img = np.pad(img, ((R, R), (R, R), (0, 0)), mode='reflect')
    pad_bg  = np.pad(bg_mask, ((R, R), (R, R)), mode='constant', constant_values=0)
    
    dist = manual_distance_transform(pad_bg)
    dist = dist[R:R+H, R:R+W] 
    
    out_bg = img.copy().astype(np.float32)
    
    for c in range(C):
        for i in range(H):
            for j in range(W):
                if bg_mask[i, j] == 0:
                    continue
                
                patch_img = pad_img[i:i+2*R+1, j:j+2*R+1, c]
                
                if dist[i, j] >= R:
                    out_bg[i, j, c] = np.sum(patch_img * h_filter)
                    
                else:
                    patch_mask = pad_bg[i:i+2*R+1, j:j+2*R+1]
                    valid_weights = h_filter * patch_mask
                    w_sum = np.sum(valid_weights)
                    
                    if w_sum > 0:
                        valid_weights /= w_sum
                        out_bg[i, j, c] = np.sum(patch_img * valid_weights)
                        
    fg_mask_3d = fg_mask[:, :, None]
    final_img = (img * fg_mask_3d) + (out_bg * (1.0 - fg_mask_3d))
    
    return np.clip(final_img, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    img_path = "deep.png"
    
    out_dir = "bokeh_outputs"
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(img_path).split('.')[0]
    
    img = plt.imread(img_path)
    if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
    if img.shape[2] == 4: img = img[:, :, :3]
    
    print("1. Creating Mask...")
    fg_mask = get_manual_mask(img)
    plt.imsave(f"{out_dir}/{base_name}_mask.png", fg_mask, cmap='gray')
    print(f"   Saved mask to {out_dir}/")
    
    print("2. Applying D=50 Bokeh...")
    bokeh_50 = apply_manual_bokeh(img, fg_mask, D=50)
    plt.imsave(f"{out_dir}/{base_name}_bokeh_D50.png", bokeh_50)
    
    print("3. Applying D=100 Bokeh...")
    bokeh_100 = apply_manual_bokeh(img, fg_mask, D=100)
    plt.imsave(f"{out_dir}/{base_name}_bokeh_D100.png", bokeh_100)
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[1].imshow(bokeh_50)
    ax[1].set_title("Bokeh D=50")
    ax[2].imshow(bokeh_100)
    ax[2].set_title("Bokeh D=100")
    
    for a in ax: a.axis('off')
    plt.tight_layout()
    
    fig.savefig(f"{out_dir}/{base_name}_comparison.png", bbox_inches='tight', dpi=300)
    
    plt.show()
