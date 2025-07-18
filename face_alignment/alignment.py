import cv2
import numpy as np
from skimage import transform as trans

# Standard landmarks for ArcFace alignment
arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

def estimate_norm(lmk, image_size=112, mode="arcface"):
    """Estimate transformation matrix for alignment"""
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0

    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]

def norm_crop(img, landmark, image_size=112, mode="arcface"):
    """Normalize and crop face image"""
    M = estimate_norm(landmark, image_size, mode)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)