import cv2
import numpy as np


def get_roi_bound(roi_points, expand_pixel=0):
    bound_min = roi_points.min(axis=0) - expand_pixel
    bound_max = roi_points.max(axis=0) + expand_pixel
    return bound_min, bound_max


def top_percent_average(flow, percent):
    r, _ = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    r = np.ravel(r)
    # r下表从小到大排序的
    idx = np.argsort(r)
    num = int(len(r) * (1 - percent))
    flow = flow.reshape((-1, 2))[idx[num:]]
    return flow.mean(axis=0)
