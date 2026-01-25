import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# =========================
# Preprocessing Image
# =========================
def preprocess_image(image, size=(128, 128)):
    """
    Resize image ke ukuran yang sama dengan training
    """
    return cv2.resize(image, size)


# =========================
# RGB Histogram Feature
# =========================
def extract_rgb_histogram(image, bins=32):
    """
    Ekstraksi fitur Histogram Warna RGB
    """
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])

    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    return np.hstack([hist_r, hist_g, hist_b])


# =========================
# LBP Feature
# =========================
def extract_lbp(image):
    """
    Ekstraksi fitur LBP (sama dengan training)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalisasi

    return hist


# =========================
# Build Feature (Warna + LBP)
# =========================
def build_feature(image):
    """
    Membangun fitur gabungan Warna + LBP
    Output shape: (1, n_features)
    """
    img = preprocess_image(image)

    color_feat = extract_rgb_histogram(img)
    lbp_feat = extract_lbp(img)

    combined_feature = np.hstack((color_feat, lbp_feat))

    return combined_feature.reshape(1, -1)
