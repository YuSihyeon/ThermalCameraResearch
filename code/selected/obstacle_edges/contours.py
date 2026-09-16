"""Conservative hysteresis: never remove a component solely for being small."""
import cv2
import numpy as np


def select_contours(probability, low=.45, high=.8):
    if not 0 <= low <= high <= 1:
        raise ValueError('Expected 0 <= low <= high <= 1')
    weak = (probability >= low).astype(np.uint8)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(count, bool)
    keep[np.unique(labels[probability >= high])] = True
    keep[0] = False
    return keep[labels].astype(np.uint8)


def thin(mask):
    return cv2.ximgproc.thinning(mask.astype(np.uint8) * 255) > 0


def edge_disagreement(first, second, valid):
    union = (first | second) & valid
    if not union.any():
        return None
    return float(((first ^ second) & valid).sum() / union.sum())


def overlay(frame, mask, background=.55):
    out = np.clip(frame.astype(np.float32) * background, 0, 255).astype(np.uint8)
    out[mask > 0] = (0, 255, 0)
    return out
