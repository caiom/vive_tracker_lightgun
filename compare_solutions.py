#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt

def morph_percentile_centroid(roi, th_bin=80, p_in=1, p_bg=99, it=2):
    roi_f = roi.astype(np.float32)
    # mask_raw = roi > th_bin

    _, mask_raw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # create nuclear interior mask (erode)
    kernel = np.ones((3,3), np.uint8)
    mask_nuc = cv2.erode(mask_raw.astype(np.uint8), kernel, iterations=2).astype(bool)
    
    # background safe mask
    mask_dil = cv2.dilate(mask_raw.astype(np.uint8), kernel, iterations=1).astype(bool)
    mask_bg = ~mask_dil
    
    if mask_nuc.sum() < 5 or mask_bg.sum() < 5:
        return np.nan, np.nan, np.zeros_like(roi_f)
    
    p_in_val = np.percentile(roi_f[mask_nuc], p_in)
    p_bg_val = np.percentile(roi_f[mask_bg], p_bg)
    
    # weight map
    weights = np.clip((roi_f - p_bg_val) / (p_in_val - p_bg_val + 1e-6), 0, 1)
    weights = mask_raw.astype(np.float32)
    
    h,w = roi.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m00 = weights.sum()
    cx = (weights * xx).sum() / m00
    cy = (weights * yy).sum() / m00
    return cx, cy, weights

# load ROI
roi = cv2.imread("C:\\Users\\Caio\\vive_tracker_lightgun\\sample_frame_0_crop2.png", cv2.IMREAD_GRAYSCALE)

# Technique 1 (morph percentile)
cx1, cy1, w_vis = morph_percentile_centroid(roi, p_in=1, p_bg=1)

# Technique 2 (gauss + sobel)
blur = cv2.GaussianBlur(roi, (5,5), 1.0)
sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sx, sy)
yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
m00_2 = sobel_mag.sum()
cx2 = (sobel_mag * xx).sum() / m00_2
cy2 = (sobel_mag * yy).sum() / m00_2

# Technique 3 (fitEllipse)
_, bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(bw, 50,150)
pts = cv2.findNonZero(edges)
if pts is not None and len(pts) >= 5:
    ellipse = cv2.fitEllipse(pts)
    (cx3, cy3), _, _ = ellipse
else:
    cx3 = cy3 = np.nan

print(f"T1 morph-mask percentile COM: cx={cx1:.3f}, cy={cy1:.3f}")
print(f"T2 Gauss+Sobel COM:           cx={cx2:.3f}, cy={cy2:.3f}")
print(f"T3 fitEllipse:                cx={cx3:.3f}, cy={cy3:.3f}")

# visualize
fig, axes = plt.subplots(1,3, figsize=(12,4))
axes[0].imshow(w_vis, cmap='gray')
axes[0].scatter(cx1, cy1, color='r')
axes[0].set_title("T1 weights (morph)")
axes[0].axis('off')
axes[1].imshow(sobel_mag.astype(np.uint8), cmap='gray')
axes[1].scatter(cx2, cy2, color='r')
axes[1].set_title("T2 Sobel")
axes[1].axis('off')
axes[2].imshow(roi, cmap='gray')
axes[2].scatter(cx3, cy3, color='r')
axes[2].set_title("T3 ellipse center")
axes[2].axis('off')
plt.tight_layout()
plt.show()
