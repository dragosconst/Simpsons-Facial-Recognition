import numpy as np
from Code.Model.sliding_window import intersection_over_union, IOU_THRESH
import matplotlib.pyplot as plt
import os
import cv2 as cv
import time
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

def compute_average_precision(rec, prec):
    m_rec = np.concatenate(([0], rec, [1]))
    m_pre = np.concatenate(([0], prec, [0]))
    for i in range(len(m_pre) - 1, -1, 1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    m_rec = np.array(m_rec)
    i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
    average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
    return average_precision

"""
Ground truth contains all annotations for every image, in (image, x1, y1, x2, y2, imclass) tuples.
"""
def evaluate_detections_facial(detections, gt, valid):
    true_detections = np.zeros(np.sum(np.asarray([len(x) for x in detections])))
    false_detections = np.zeros(np.sum(np.asarray([len(x) for x in detections])))
    gt_exists_detection = np.zeros(len(gt))

    d_index = 0
    for im_index, detection in enumerate(detections):
        for box in detection:
            max_overlap = -1
            max_index = None
            for gt_index, tuples in enumerate(gt):
                g_index, *g_box, im_class = tuples
                if g_index < im_index:
                    continue
                if g_index > im_index:
                    break
                # here we have one of the true detections, let's check it against the current detection
                overlap = intersection_over_union(box, g_box)
                if overlap > max_overlap:
                    # print(g_index, g_box, im_class)
                    xd1, yd1, xd2, yd2 = box
                    x1, y1, x2, y2 = g_box
                    # array_to_img(valid[im_index][y1:y2, x1:x2]).show()
                    # array_to_img(valid[im_index][yd1:yd2, xd1:xd2]).show()
                    # time.sleep(3)
                    max_overlap = overlap
                    max_index = gt_index
            if max_overlap >= IOU_THRESH:
                print("huh")
                if gt_exists_detection[max_index] == 0:
                    print("bruuuh")
                    true_detections[d_index] = 1
                    gt_exists_detection[max_index] = 1
                else:
                    false_detections[d_index] = 1
            else:
                false_detections[d_index] = 1
            d_index += 1

    cum_false_positive = np.cumsum(false_detections)
    cum_true_positive = np.cumsum(true_detections)

    rec = cum_true_positive / len(valid)
    print(rec, cum_true_positive, len(valid))
    prec = cum_true_positive / (cum_true_positive + cum_false_positive)
    average_precision = compute_average_precision(rec, prec)
    plt.plot(rec, prec, '-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average precision %.3f' % average_precision)
    plt.savefig(os.path.join("data/", 'precizie_medie.png'))
    plt.show()