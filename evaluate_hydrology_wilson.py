import os
import math
import cv2
import numpy as np

def wilson_interval(k, n, confidence=0.95):
    """Calculates Wilson score confidence intervals."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996  # 95% Confidence level
    p = k / n
    denom = 1 + (z**2) / n
    centre = p + (z**2) / (2 * n)
    spread = math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)
    lower = max(0.0, (centre - z * spread) / denom)
    upper = min(1.0, (centre + z * spread) / denom)
    return p, lower, upper

def classify_pothole_crop(crop_rgb):
    h, w, _ = crop_rgb.shape
    if h <= 8 or w <= 8:
        return False
    crop_gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    crop_hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    
    gx = cv2.Sobel(crop_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Criterion 1: Specular Skylight Mirroring (High Lux, Low Saturation, Low Roughness)
    specular_mask = (crop_gray > 165) & (crop_hsv[:, :, 1] < 45) & (grad_mag < 22)
    spec_ratio = np.sum(specular_mask) / (crop_gray.size + 1e-5)

    # Criterion 2: Dark Liquid Core Absorption
    dark_water_mask = (crop_gray < 65) & (crop_hsv[:, :, 1] < 60) & (grad_mag < 18)
    dark_ratio = np.sum(dark_water_mask) / (crop_gray.size + 1e-5)

    # Spatial Coherence Blob Check
    if spec_ratio >= 0.08:
        n, _, stats, _ = cv2.connectedComponentsWithStats(specular_mask.astype(np.uint8))
        if n > 1 and np.max(stats[1:, cv2.CC_STAT_AREA]) >= 0.05 * crop_gray.size:
            return True
            
    if dark_ratio >= 0.20:
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark_water_mask.astype(np.uint8))
        if n > 1 and np.max(stats[1:, cv2.CC_STAT_AREA]) >= 0.12 * crop_gray.size:
            return True
            
    return False

def run_evaluation(data_dir="test_hydrology"):
    tp, fp, tn, fn = 0, 0, 0, 0
    categories = [("wet", True), ("dry", False)]
    
    for folder_name, true_label in categories:
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_bgr = cv2.imread(os.path.join(folder_path, fname))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pred_wet = classify_pothole_crop(img_rgb)
            
            if true_label and pred_wet:
                tp += 1
            elif true_label and not pred_wet:
                fn += 1
            elif not true_label and not pred_wet:
                tn += 1
            elif not true_label and pred_wet:
                fp += 1

    total = tp + fp + tn + fn
    if total == 0:
        print("\n[!] Directory is empty. Place test images into:")
        print("    - test_hydrology/wet/")
        print("    - test_hydrology/dry/")
        return

    acc, acc_l, acc_h = wilson_interval(tp + tn, total)
    prec, prec_l, prec_h = wilson_interval(tp, tp + fp) if (tp + fp) > 0 else (0, 0, 0)
    rec, rec_l, rec_h = wilson_interval(tp, tp + fn) if (tp + fn) > 0 else (0, 0, 0)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print("\n" + "="*58)
    print("       HYDROLOGY VALIDATION RESULTS (WILSON INTERVALS)")
    print("="*58)
    print(f"Sample Count : N = {total} (TP={tp}, FP={fp}, TN={tn}, FN={fn})")
    print(f"Accuracy     : {acc*100:5.2f}%  [95% CI: {acc_l*100:.1f}% - {acc_h*100:.1f}%]")
    print(f"Precision    : {prec*100:5.2f}%  [95% CI: {prec_l*100:.1f}% - {prec_h*100:.1f}%]")
    print(f"Recall       : {rec*100:5.2f}%  [95% CI: {rec_l*100:.1f}% - {rec_h*100:.1f}%]")
    print(f"F1-Score     : {f1*100:5.2f}%")
    print("="*58)

if __name__ == "__main__":
    run_evaluation()