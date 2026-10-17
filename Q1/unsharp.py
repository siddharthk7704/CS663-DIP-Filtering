import numpy as np
import matplotlib.pyplot as plt

def get_separable_gaussian(size=5, sigma=1.0):
    center = size // 2
    x = np.arange(-center, center + 1)
    
    g_1d = np.exp(-(x**2) / (2 * sigma**2))
    g_1d = g_1d / np.sum(g_1d)
    
    U = g_1d.reshape(-1, 1)
    V = g_1d.reshape(1, -1)
    
    return U, V

def loop_convolution_separable(image, U, V):
    img_float = image.astype(np.float32)
    img_h, img_w = img_float.shape
    k_size = U.shape[0]
    pad = k_size // 2
    
    pad_vert = np.pad(img_float, ((pad, pad), (0, 0)), mode='edge')
    temp_img = np.zeros_like(img_float)
    
    for i in range(img_h):
        for j in range(img_w):
            patch = pad_vert[i : i + k_size, j : j + 1]
            temp_img[i, j] = np.sum(patch * U)
            
    pad_horiz = np.pad(temp_img, ((0, 0), (pad, pad)), mode='edge')
    blurred_img = np.zeros_like(img_float)
    
    for i in range(img_h):
        for j in range(img_w):
            patch = pad_horiz[i : i + 1, j : j + k_size]
            blurred_img[i, j] = np.sum(patch * V)
            
    return blurred_img

def compute_blur(image, size=5, sigma=1.0):
    img_float = image.astype(np.float32)
    U_vec, V_vec = get_separable_gaussian(size, sigma)
    
    blurred_F = np.zeros_like(img_float)
    
    if img_float.ndim == 3:
        if img_float.shape[2] == 4:
            img_float = img_float[:, :, :3]
            
        for c in range(3):
            blurred_F[:, :, c] = loop_convolution_separable(img_float[:, :, c], U_vec, V_vec)
    else:
        blurred_F = loop_convolution_separable(img_float, U_vec, V_vec)
        
    return blurred_F

def apply_unsharp_mask(image, blurred_image, s=1.0):
    img_float = image.astype(np.float32)
    
    if img_float.ndim == 3 and img_float.shape[2] == 4:
        img_float = img_float[:, :, :3]
        
    mask = img_float - blurred_image
    sharpened = img_float + (s * mask)
        
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    sharpened_clipped = np.clip(sharpened, min_intensity, max_intensity)
    
    return sharpened_clipped.astype(image.dtype)

image_path = "tiger.png"
original_img = plt.imread(image_path)

if np.max(original_img) <= 1.0:
    original_img = (original_img * 255).astype(np.float32)

blurred_img = compute_blur(original_img, size=5, sigma=2.0)

s_moderate = 3.0
s_strong = 8.0

img_sharp_1 = apply_unsharp_mask(original_img, blurred_img, s=s_moderate)
img_sharp_2 = apply_unsharp_mask(original_img, blurred_img, s=s_strong)

cmap_setting = 'gray' if original_img.ndim == 2 else None
plt.imsave("tiger_sharpened_moderate.png", img_sharp_1.astype(np.uint8), cmap=cmap_setting)
plt.imsave("tiger_sharpened_strong.png", img_sharp_2.astype(np.uint8), cmap=cmap_setting)

min_val, max_val = np.min(original_img), np.max(original_img)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_img.astype(np.uint8), cmap='gray', vmin=min_val, vmax=max_val)
axes[0].set_title("1. Original Image")
axes[0].axis('off')

axes[1].imshow(img_sharp_1.astype(np.uint8), cmap='gray', vmin=min_val, vmax=max_val)
axes[1].set_title(f"2. Sharpened (Moderate: s={s_moderate})")
axes[1].axis('off')

axes[2].imshow(img_sharp_2.astype(np.uint8), cmap='gray', vmin=min_val, vmax=max_val)
axes[2].set_title(f"3. Sharpened (Strong: s={s_strong})")
axes[2].axis('off')

plt.tight_layout()
plt.show()
