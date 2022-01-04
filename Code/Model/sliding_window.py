import numpy as np
import cv2 as cv
from Code.Model.cnn_bow import extract_sift_features_image, K
from scipy.cluster.vq import vq
from Code.Logical.classes import FaceClasses

IOU_THRESH = 0.3
MAX_IMG_WIDTH = 400
MAX_IMG_HEIGHT = 400
FAC_RESIZE = 2
SCALE_FACTOR = 0.2
STRIDE_Y = 2
STRIDE_X = 2
SW_W = 32
SW_H = 32

def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)


    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)

    return iou

"""
Params:
- image_detections: np array of (x1, y1, x2, y2) tuples
- image_scores: np array of scores (with Svm it will be 1 everywhere i guess)
- image_size: shape of original image
This is called for one image at a time
"""
def non_maximal_suppression(image_detections, image_scores, image_size):
    if len(image_detections) == 0:
        return [], None
    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
    print(x_out_of_bounds, y_out_of_bounds)
    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]
    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = IOU_THRESH
    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True: # don't change to 'is True' because is a numpy True and is not a python True :)
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j] == True: # don't change to 'is True' because is a numpy True and is not a python True :)
                    if intersection_over_union(sorted_image_detections[i],
                                                    sorted_image_detections[j]) > iou_threshold:
                        is_maximal[j] = False
                    else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                            is_maximal[j] = False


    return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

# check it can actually detect any faces at all?? lol
def check_detections_directly(valid_data, valid_labels, code_book, svm):
    sift = cv.SIFT_create(edgeThreshold=15, contrastThreshold=0.03)
    kps = []
    for im_index, image in enumerate(valid_data):
        kp = sift.detect(image, None)
        kps.append(kp)
    kps, dp = sift.compute(valid_data, kps)
    histograms = np.zeros((len(valid_data), K), np.float32)
    for im_index, image in enumerate(valid_data):
        vwords, distances = vq(dp[im_index], code_book)
        for vw in vwords:
            histograms[im_index, vw] += 1

    print(f"Score on real data is {svm.score(histograms, valid_labels)}")


"""
I will resize the image several times and apply the sliding window over each image. I will use a semi-smart
resizing, meaning that I have a max bound for resizing, and no min bound -  I just keep making it smaller
untill it gets smaller than the sliding windows times a factor.
The final coordinates returned will obviously be returned for the initial image.
"""
def sliding_window_valid(valid_data, code_book, svm):
    all_detections = []
    for img_index, image in enumerate(valid_data):
        h, w, _ = image.shape
        scale = min(MAX_IMG_HEIGHT / h, MAX_IMG_WIDTH / w)
        detections = []
        # go from highest possible scaling to smallest scaling
        while h * scale >= FAC_RESIZE * SW_H and w * scale >= FAC_RESIZE * SW_H:
            img = cv.resize(image, (0, 0), fx=scale, fy=scale)

            # apply sliding window on img
            for y in range(0, img.shape[0] - SW_H + 1, STRIDE_Y):
                # print(f"y is {y}, stop at {img.shape[0] - SW_H + 1}")
                for x in range(0, img.shape[1] - SW_W + 1, STRIDE_X):
                    patch = img[y:y+SW_H, x:x+SW_W]

                    # extract histogram of patch
                    histogram = np.zeros(K)
                    descriptors = extract_sift_features_image(patch)
                    vwords, distances = vq(descriptors, code_book)
                    for vw in vwords:
                        histogram[vw] += 1

                    predicted_label = svm.predict([histogram])[0]
                    cv.imshow("patch", patch)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                    if predicted_label == FaceClasses.Face: # scale detection back to original scale
                        print("face detected")
                        detections.append((x // scale, y // scale, x // scale + SW_W // scale, y // scale + SW_H // scale))


            # print(f"Scale is now {scale}, width is {w * scale}, height is {h * scale}")
            scale = scale - scale * SCALE_FACTOR

        # now we have to filter out non-maximal detections
        print(f"Finished image {img_index} out of {len(valid_data)} images.")
        scores = np.ones(len(detections))
        detections, _ = non_maximal_suppression(np.asarray(detections), scores, image.shape)
        all_detections.append([detections])

    return all_detections