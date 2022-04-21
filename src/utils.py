import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import scipy


def readTruths(filepath):
    true_boxes = []
    for line in open(filepath, "r").readlines():
        vertices = line.split(",")
        box = np.int32([[vertices[0], vertices[1]], [vertices[2], vertices[3]], [vertices[4], vertices[5]], [vertices[6], vertices[7]]])
        true_boxes.append(box)

    return np.array(true_boxes)


def union_iou(boxes, true_boxes):
    true_polygon = unary_union([Polygon(box) for box in true_boxes])
    detected_polygon = unary_union([Polygon(box) for box in boxes])

    union = unary_union([true_polygon, detected_polygon])

    if union.area == 0:
        return 0
    else:
        intersection = make_valid(detected_polygon.intersection(true_polygon))
        return intersection.area / union.area

def bbox_iou(boxA, boxB):
    # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # ^^ corrected.

    true_polygon = Polygon(boxA)
    detected_polygon = Polygon(boxB)

    union = unary_union([true_polygon, detected_polygon])

    if union.area == 0:
        return 0
    else:
        intersection = make_valid(detected_polygon.intersection(true_polygon))
        return intersection.area / union.area


def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2].
      The number of bboxes, N1 and N2, need not be the same.

    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i, :], bbox_pred[j, :])

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((diff, n_pred), MIN_IOU)),
                                    axis=0)

    if n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((n_true, diff), MIN_IOU)),
                                    axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label


