# shared/vision.py
import cv2
import numpy as np

def preprocess_frame_rgb(frame_rgb, img_w, img_h):
    """
    frame_rgb: H x W x 3 uint8 RGB
    returns: (3, img_h, img_w) float32 in [0,1]
    """
    img = cv2.resize(frame_rgb, (img_w, img_h))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img