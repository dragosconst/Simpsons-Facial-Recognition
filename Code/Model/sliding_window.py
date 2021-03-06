import numpy as np
import cv2 as cv
from Code.Model.cnn_bow import extract_sift_features_image, K
from scipy.cluster.vq import vq
from Code.Logical.classes import FaceClasses
from Code.IO.load_data import FACE_WIDTH, FACE_HEIGHT, IM_WIDTH, IM_HEIGHT
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import VGG19, vgg19
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from Code.Data_Processing.yellow_filter import apply_filters

IOU_THRESH = 0.3
MAX_IMG_WIDTH = 140
MAX_IMG_HEIGHT = 140
DESIRED_WIDTH = 128
DESIRED_HEIGHT = 256
FAC_RESIZE = 1
SCALE_FACTOR = 0.2
MIN_SCALE = 0.7
MAX_SCALE = 3
DIV_SCALE = 1.42
MIN_NEIGH = 0.5
STRIDE_Y = 2
STRIDE_X = 2
SW_W = 26
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
def non_maximal_suppression(image_detections, image_scores, image_size, scores_extra=None):
    if len(image_detections) == 0:
        if scores_extra is None:
            return np.array([]), np.array([])
        return np.array([]), np.array([]), np.array([])
    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
    print(x_out_of_bounds, y_out_of_bounds)
    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]
    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]
    if scores_extra is not None:
        sorted_exra = scores_extra[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = IOU_THRESH
    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True: # don't change to 'is True' because is a numpy True and is not a python True :)
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j] == True: # don't change to 'is True' because is a numpy True and is not a python True :)
                    if intersection_over_union(sorted_image_detections[i],
                                                    sorted_image_detections[j]) > iou_threshold \
                        or is_child(sorted_image_detections[j], sorted_image_detections[i]):
                        is_maximal[j] = False
                    else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                            is_maximal[j] = False


    if scores_extra is None:
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]
    return sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_exra[is_maximal]

def is_child(child, parent, to_print=False):
    x_a = max(child[0], parent[0])
    y_a = max(child[1], parent[1])
    x_b = min(child[2], parent[2])
    y_b = min(child[3], parent[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)


    child_area = (child[2] - child[0] + 1) * (child[3] - child[1] + 1)
    parent_area = (parent[2] - parent[0] + 1) * (parent[3] - parent[1] + 1)

    if to_print:
        print(inter_area / child_area)
    return inter_area / child_area >= 0.3

# check it can actually detect any faces at all?? lol
def check_detections_directly(valid_data, valid_labels, code_book, classifier, scaler):
    valid_data = vgg19.preprocess_input(valid_data)
    vgg = VGG19(include_top=False, input_shape=(FACE_HEIGHT, FACE_WIDTH, 3))
    valid_features = vgg.predict(valid_data)
    print(f"Score on real data is {classifier.evaluate(valid_features, valid_labels, verbose=1)}")

def check_neighbours(image, x1, y1, sw_w, sw_h, classifier, vgg):
    slide_factor = 0.15
    up = image[max(0, int(y1 - slide_factor * sw_h)):max(0, int(y1 - slide_factor * sw_h) + sw_h), x1:x1+sw_w]
    up = img_to_array(array_to_img(up).resize((FACE_WIDTH, FACE_HEIGHT)))
    down = image[min(int(y1 + slide_factor * sw_h), image.shape[0]):min(int(y1 + slide_factor * sw_h) + sw_h, image.shape[0]), x1:x1+sw_w]
    down = img_to_array(array_to_img(down).resize((FACE_WIDTH, FACE_HEIGHT)))
    left = image[y1:y1+sw_h, max(0, int(x1 - slide_factor * sw_w)):max(0, int(x1 - slide_factor * sw_w)+sw_w)]
    left = img_to_array(array_to_img(left).resize((FACE_WIDTH, FACE_HEIGHT)))
    right = image[y1:y1+sw_h, min(int(x1 + slide_factor * sw_w), image.shape[1]):min(int(x1 + slide_factor * sw_w)+sw_w, image.shape[1])]
    right = img_to_array(array_to_img(right).resize((FACE_WIDTH, FACE_HEIGHT)))
    neighbours = np.array([up, down, left, right])
    neighbours = vgg19.preprocess_input(neighbours)
    neighbours_features = vgg.predict(neighbours)
    neighbours_classes = classifier.predict(neighbours_features)
    # print(np.argmin(neighbours_classes[:, 0]))
    return np.min(neighbours_classes[:, 0]) # return neighbour that looks least like a face


"""
I will resize the image several times and apply the sliding window over each image. I will use a semi-smart
resizing, meaning that I have a max bound for resizing, and no min bound -  I just keep making it smaller
untill it gets smaller than the sliding windows times a factor.
The final coordinates returned will obviously be returned for the initial image.
"""
def sliding_window_valid(valid_data, classifier, faces_classifier):
    all_detections = []
    all_scores = []
    all_faces_detections = []
    all_faces_scores = []
    vgg = VGG19(include_top=False, input_shape=(FACE_HEIGHT, FACE_WIDTH, 3))
    for img_index, image in enumerate(valid_data):
        h, w, _ = image.shape
        scale_image = min(MAX_IMG_HEIGHT / h, MAX_IMG_WIDTH / w)
        image = img_to_array(array_to_img(image).resize((int(w * scale_image), int(h * scale_image))))
        # scalew = DESIRED_WIDTH / w
        # scaleh = DESIRED_HEIGHT / h
        detections = []
        faces_detections = []
        scores = []
        scores_detections = []
        # go from highest possible scaling to smallest scaling
        image_masked = apply_filters(image)
        # array_to_img(image).show()
        scale = MAX_SCALE
        while scale >= MIN_SCALE:
            print(scale)
            sw_h = int(SW_H * scale)
            sw_w = int(SW_W * scale)
            # apply sliding window on img
            for y in range(0, image.shape[0] - sw_h + 1, STRIDE_Y):
                for x in range(0, image.shape[1] - sw_w + 1, STRIDE_X):
                    patch = image[y:int(y+sw_h), x:int(x+sw_w)]
                    # old_patch = patch.copy()
                    patch_masked = image_masked[y:int(y+sw_h), x:int(x+sw_w)]
                    if (np.sum(patch_masked[:3, :]) <= 2 or np.sum(patch_masked[:, :3]) <= 2) or np.mean(patch_masked) <= 2:
                        continue
                    patch = img_to_array(array_to_img(patch).resize((FACE_WIDTH, FACE_HEIGHT)))

                    # histogram = scaler.transform([histogram])[0]
                    patch = vgg19.preprocess_input(patch)
                    features = vgg.predict(np.asarray([patch]))[0]

                    predicted_label = classifier.predict_classes(np.asarray([features]))[0]
                    # if scale <= 0.5:
                    #     print(y, x)
                    #     cv.imshow('peci', cv.cvtColor(np.array(array_to_img(old_patch)), cv.COLOR_RGB2BGR))
                    #     cv.waitKey(0)
                    #     cv.destroyAllWindows()
                    if predicted_label == FaceClasses.Face.value\
                            and check_neighbours(image, x,y, sw_w, sw_h, classifier, vgg) >= MIN_NEIGH: # scale detection back to original scale
                        # print("face detected")
                        # scores.append(np.max(classifier.predict(np.asarray([features]))[0]))
                        scores.append(check_neighbours(image, x, y, sw_w, sw_h, classifier, vgg))
                        # print(np.max(classifier.predict(np.asarray([features]))[0]))
                        # if img_index == 1:
                        # array_to_img(old_patch).show(title='face')
                        # array_to_img(image[int(y / scale):int((y + SW_H) / scale), int(x / scale):int((x  + SW_W) / scale)]).show(title='original')
                        detections.append((x, y, x + sw_w, y + sw_h, img_index))
                        face_class = faces_classifier.predict_classes(np.asarray([features]))[0]
                        scores_detections.append(np.max(faces_classifier.predict(np.asarray([features]))[0]))
                        faces_detections.append((x, y, x + sw_w, y + sw_h, face_class, img_index))

            # detections, scores = non_maximal_suppression(np.asarray(detections), np.asarray(scores, np.float32), image.shape)
            # detections = list(detections)
            # scores = list(scores)
            # scale = scale - scale * SCALE_FACTOR
            scale /= DIV_SCALE
            # scaleh = scaleh - scaleh * SCALE_FACTOR
            # scalew = scalew - scalew * SCALE_FACTOR

        # now we have to filter out non-maximal detections
        print(f"Finished image {img_index} out of {len(valid_data)} images.")
        scores = np.asarray(scores, np.float32)
        old_scores = scores.copy()
        detections, scores = non_maximal_suppression(np.asarray(detections), scores, image.shape)
        scores_detections = np.asarray(scores_detections, np.float32)
        faces_detections, _, scores_detections = non_maximal_suppression(np.asarray(faces_detections), old_scores, image.shape, scores_detections)
        detections = detections.astype(np.int32)
        faces_detections = faces_detections.astype(np.int32)
        good_indexes = np.ones(len(detections)).astype(bool)
        for d_index, detection in enumerate(detections):
            if good_indexes[d_index] == False:
                continue
            for d_index2, detection2 in enumerate(detections[d_index:]):
                d_index2 += d_index
                if d_index2 == d_index:
                    continue
                if good_indexes[d_index2] == False:
                    continue
                print(d_index, d_index2)
                if is_child(detection, detection2, to_print=True):
                    good_indexes[d_index] = False
                elif is_child(detection2, detection, to_print=True):
                    good_indexes[d_index2] = False
        detections = detections[good_indexes]
        scores = scores[good_indexes]
        faces_detections = faces_detections[good_indexes]
        scores_detections = scores_detections[good_indexes]

        # for x1, y1, x2, y2, _ in detections:
        #     array_to_img(image[y1:y2, x1:x2, :]).show()
        for index, detection in enumerate(detections):
            x1, y1, x2, y2, i_index = detection
            detections[index] = (int(x1 / scale_image), int(y1 / scale_image), int(x2 / scale_image), int(y2 / scale_image), i_index)
        for index, detection in enumerate(faces_detections):
            *coords, face, i_index = faces_detections[index]
            x1, y1, x2, y2 = coords
            faces_detections[index] = (int(x1 / scale_image), int(y1 / scale_image), int(x2 / scale_image), int(y2 / scale_image), face, i_index)

        all_detections += list(detections)
        all_scores += list(scores)
        all_faces_detections += list(faces_detections)
        all_faces_scores += list(scores_detections)

    return np.asarray(all_detections), np.asarray(all_scores), np.asarray(all_faces_detections), np.asarray(all_faces_scores)