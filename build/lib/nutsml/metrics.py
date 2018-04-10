"""
.. module:: metrics
   :synopsis: Evaluation metrics for model performance
"""
from __future__ import absolute_import


def box_intersect(box1, box2):
    """
    Return intersection box two (bounding) boxes.

    If the boxes don't intersect (0,0,0,0) is returned.

    :param (x,y,w,h) box1: First box with x,y specifying the left upper corner.
    :param (x,y,w,h) box2: Second bounding box.
    :return: Intersection box
    :rtype: (x,y,w,h)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    w, h = x2 - x1, y2 - y1
    return (x1, y1, w, h) if w > 0 and h > 0 else (0, 0, 0, 0)


def box_iou(box1, box2):
    """
    Return Intersection of Union of two (bounding) boxes.

    :param (x,y,w,h) box1: First bounding box with x,y specifying the
            left upper corner.
    :param (x,y,w,h) box2: Second bounding box.
    :return: Intersection of Union [0...1]
    :rtype: float
    """
    _, _, w, h = box_intersect(box1, box2)
    area_intersect = w * h
    area_box1 = box1[2] * box1[3]
    area_box2 = box2[2] * box2[3]
    return area_intersect / float(area_box1 + area_box2 - area_intersect)


def box_matches(box, boxes):
    """
    Return list of boxes and their IOU's sorted by maximum overlap first.

    :param [x,y,w,h] box: Box to match against all boxes.
    :param list boxes: Boxes that are overlapped with box. Boxes must be
       in format [[x,y,w,h], ...]
    :return: List of boxes with their IOUs: [ (box, IOU), ... ]
    :rtype: list of tuples
    """
    matches = [(b, box_iou(box, b)) for b in boxes]
    matches.sort(key=lambda m: -m[1])
    return matches


def box_best_match(box, boxes):
    """
    Return box in boxes that has the largest IOU with the given box.

    :param [x,y,w,h] box: Box to match against all boxes.
    :param list boxes: Boxes that are overlapped with box. Boxes must be
       in format [[x,y,w,h], ...]
    :return: Best matching box in format ([x,y,w,h], iou) or (None, 0.0)
      if boxes is empty.
    """
    if not boxes:
        return None, 0.0
    return max(((b, box_iou(box, b)) for b in boxes), key=lambda m: m[1])


def box_pr_curve(true_boxes, pred_boxes, pred_scores, iou_thresh=0.5):
    """
    Return Precision Recall curve for bounding box predictions.

    References:
    https://sanchom.wordpress.com/tag/average-precision

    :param list true_boxes: Collection of true boxes of format (x,y,w,h).
    :param list pred_boxes: Collection of predicted boxes of format (x,y,w,h).
    :param list pred_scores: Scores/confidence values for each predicted box.
       Higher score means more confident.
    :param float iou_thresh: Threshold for minimum IoU overlap for a box to
       be counted as a true positive.
    :return: Precision-recall curve
    :rtype: generator over tuples (precision, recall, score)
    :raises: AssertionError if pred_boxes and pred_scores don't match in length.
    """
    assert len(pred_scores) == len(pred_boxes)
    if true_boxes and pred_boxes:
        tp, fp = 0, 0
        true_unmatched = {tuple(b) for b in true_boxes}
        preds = sorted(zip(pred_boxes, pred_scores), key=lambda s: -s[1])
        for pred_box, pred_score in preds:
            box, iou = box_best_match(pred_box, true_unmatched)
            if iou >= iou_thresh:
                true_unmatched.discard(box)
                tp += 1
            else:
                fp += 1
            fn = len(true_unmatched)
            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)
            yield precision, recall, pred_score


def box_avg_precision(true_boxes, pred_boxes, pred_scores, iou_thresh=0.5,
                      mode='voc2007'):
    """
    Return Average Precision (AP).

    Note that there are various, different definitions of Average Precision.
    See: https://arxiv.org/abs/1607.03476
    and: https://sanchom.wordpress.com/tag/average-precision/
    This code implements the version used in VOC challenge 2007, see
    | http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
    | http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf

    :param list true_boxes: Collection of true boxes of format (x,y,w,h).
    :param list pred_boxes: Collection of predicted boxes of format (x,y,w,h).
    :param list pred_scores: Scores/confidence values for each predicted box.
       Higher score means more confident.
    :param float iou_thresh: Threshold for minimum IoU overlap for a box to
       be counted as a true positive.
    :param string mode: Currently not used and set to 'voc2007'
    :return: Average Precision [0...1]
    :rtype: float
    :raises: AssertionError if pred_boxes and pred_scores don't match in length.
    """
    pr_curve = box_pr_curve(true_boxes, pred_boxes, pred_scores, iou_thresh)
    pr_curve = list(pr_curve)  # generator to list
    if not pr_curve:
        return 0.0
    rts = [r / 10.0 for r in range(0, 11)]  # recall thresholds

    def p_inter(rt):
        ps = [p for p, r, _ in pr_curve if r >= rt]
        return max(ps) if ps else 0

    return sum(p_inter(rt) for rt in rts) / 11


def box_mean_avg_precision(true_boxes, true_labels, pred_boxes, pred_labels,
                           pred_scores, iou_thresh=0.5, mode='voc2007'):
    """
    Return mean Average Precision (mAP).

    See box_avg_precision() for details.

    :param list true_boxes: Collection of true boxes of format (x,y,w,h).
    :param list true_labels: Label for each true box.
    :param list pred_boxes: Collection of predicted boxes of format (x,y,w,h).
    :param lsit pred_labels: Label for each predicted box.
    :param list pred_scores: Scores/confidence values for each predicted box.
       Higher score means more confident.
    :param float iou_thresh: Threshold for minimum IoU overlap for a box to
       be counted as a true positive.
    :param string mode: Currently not used and set to 'voc2007'
    :return: mean Average Precision [0...1]
    :rtype: float
    :raises: AssertionError if pred_boxes and pred_scores don't match in length.
    """
    assert len(true_boxes) == len(true_labels)
    assert len(pred_boxes) == len(pred_labels)
    assert len(pred_boxes) == len(pred_scores)
    labels = set(true_labels + pred_labels)
    sap, n = 0.0, 0
    for label in labels:
        preds = zip(pred_boxes, pred_labels, pred_scores)
        pbss = [(b, s) for b, l, s in preds if l == label]
        tbs = [b for b, l in zip(true_boxes, true_labels) if l == label]
        if tbs and pbss:
            pbs, pss = zip(*pbss)
            sap += box_avg_precision(tbs, pbs, pss, iou_thresh, mode=mode)
            n += 1
    return sap / n if n else 0
