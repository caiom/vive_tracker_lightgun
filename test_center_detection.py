#%%
"""
Detecção sub-pixel de marcadores retrorrefletivos com visualização etapa a etapa.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# ---------- núcleo do processamento -----------------------------------------
def process_image(img_bgr, K=None, D=None,
                  min_area=20, max_area=20000):
    """Retorna (centros Nx2, dicionário de imagens de debug)."""
    dbg = {}                                       # imagens p/ plot
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dbg['gray'] = gray

    # Remover distorção se fornecido
    if K is not None and D is not None:
        h, w = gray.shape
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h),
                                                 cv2.CV_32FC1)
        gray = cv2.remap(gray, map1, map2, cv2.INTER_LINEAR)
        dbg['undistorted'] = gray

    # # Flat-field simples (compensa LEDs centrais)
    # blur = cv2.medianBlur(gray, 31)
    # norm = cv2.divide(gray, blur, scale=255)
    # dbg['flatfield'] = norm

    # Binarização de Otsu
    _, bw = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dbg['binary'] = bw

    # Limpeza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw_clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=2)
    dbg['morph'] = bw_clean

    # Achar contornos
    contours, _ = cv2.findContours(bw_clean, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    centers = []
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if len(cnt) < 5 or not (min_area < area < max_area):
            continue

        # Ajuste de elipse + centro sub-pixel
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        # win  = (5, 5)
        # crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        # pt   = cv2.cornerSubPix(gray.astype(np.float32),
        #                         np.array([[cx, cy]], np.float32),
        #                         win, (-1, -1), crit)
        # cx, cy = pt[0]
        centers.append([cx, cy])

        # desenha para debug
        cv2.ellipse(overlay, ((cx, cy), (MA, ma), angle), (0, 255, 0), 2)
        cv2.circle(overlay, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    dbg['fit'] = overlay
    return np.array(centers, float), dbg


# ---------- plotting ---------------------------------------------------------
def show_stages(dbg_dict, title=''):
    """Mostra as seis etapas principais em um único figure."""
    stages = ['gray', 'flatfield', 'binary', 'morph', 'fit']
    stages = ['gray', 'binary', 'morph', 'fit']
    names  = ['Grayscale',
            #   'Flat-field Normalized',
              'Otsu Binarized',
              'After Morphology',
              'Ellipse Fit w/ Centers']
    n = len(stages)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5*rows), dpi=900)
    fig.suptitle(title, fontsize=16)
    for idx, key in enumerate(stages):
        r, c = divmod(idx, cols)
        ax = axs[r, c] if rows > 1 else axs[c]
        img = dbg_dict[key]
        cmap = 'gray' if len(img.shape) == 2 else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(names[idx])
        ax.axis('off')

    # esconde eixos vazios
    for j in range(n, rows*cols):
        r, c = divmod(j, cols)
        ax = axs[r, c] if rows > 1 else axs[c]
        ax.axis('off')
    plt.tight_layout()
    plt.show()



fp = "C:\\Users\\Caio\\vive_tracker_lightgun\\sample_frame_0.png"

K = "C:\\Users\\Caio\\vive_tracker_lightgun\\calib_images_icam_8mm_2\\cam_matrix.npy"
D = "C:\\Users\\Caio\\vive_tracker_lightgun\\calib_images_icam_8mm_2\\distortion.npy"

K = np.load(K)
D = np.load(D)

img = cv2.imread(fp, cv2.IMREAD_COLOR)

centers, dbg = process_image(img, K, D)
print(f'\nArquivo: {fp}')
for i, (x, y) in enumerate(centers):
    print(f'  marcador {i:02d}: (x={x:.3f}, y={y:.3f}) px')
show_stages(dbg, title=Path(fp).name)
