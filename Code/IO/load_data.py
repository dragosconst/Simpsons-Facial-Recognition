import os
import numpy as np
from glob import glob
import cv2 as cv
from Code.Logical.classes import ImageClasses, FaceClasses
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

TRAIN_PATH = "antrenare/"
VALID_PATH = "validare/"
VALID_IM_PATH = "simpsons_validare"
DATA_PATH = "data/"
FACES = "faces"
FACES_LAB = "faces_labeled"
VALID = "valid"
VALID_LAB = "valid_lab"
NEG = "negative"
NAMES = ["bart", "homer", "lisa", "marge"]
IMGS_PER_CHAR = 1101  # holds true for all names
CHARS = 4
IMGS = CHARS * IMGS_PER_CHAR
IM_WIDTH = 128
IM_HEIGHT = 256
FACE_WIDTH = 48
FACE_HEIGHT = 48
NEG_EX_PER_IMG = 10

"""
Will load all data in a gigantic array of images. Labels are loaded in a (x1, y1, x2, y2, c) tuple, where c is the class
type
"""

# check if this line belongs to the given pic
def match_beginning_of_line(line, pic_index):
    STARTING_TEXT_LEN = len("pic_")
    for index, char in enumerate(line):
        if index < STARTING_TEXT_LEN:
            continue
        number = int(line[index:index + 4])
        if number == pic_index:
            return True
        return False

def get_data_from_line(line):
    SKIP_CHARS = len("pic_0000.jpg ")
    relevant_line = line[SKIP_CHARS:].split(" ")
    im_class = relevant_line[4]
    if im_class == "bart":
        im_class = ImageClasses.Bart.value
    elif im_class == "homer":
        im_class = ImageClasses.Homer.value
    elif im_class == "marge":
        im_class = ImageClasses.Marge.value
    elif im_class == "lisa":
        im_class = ImageClasses.Lisa.value
    elif im_class == "unknown":
        im_class = ImageClasses.Unknown.value
    return int(relevant_line[0]), int(relevant_line[1]), int(relevant_line[2]), int(relevant_line[3]), im_class

def load_raw_imgs():
    faces_labeled = []
    faces = []
    bad_faces = []
    for name in NAMES:
        imgs_path = os.path.join(TRAIN_PATH + name + "/", "*.jpg")
        gt_path = os.path.join(TRAIN_PATH, name + '.txt')
        gt_file = open(gt_path, "rt")
        gt_lines = gt_file.readlines()
        line_index = 0
        files = glob(imgs_path)
        for index, file in enumerate(files):
            img = Image.open(file)
            img = img_to_array(img)
            # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            oldw, oldh = img.shape[1], img.shape[0]
            # generate positive examples
            valid_pixels = [[i * img.shape[0] + j for j in range(img.shape[1])] for i in range(img.shape[0])]
            valid_pixels = np.array(valid_pixels)
            valid_pixels[-FACE_HEIGHT:, :] = -1 # so that i don't choose negative examples too close to the edges
            valid_pixels[:, -FACE_WIDTH:] = -1
            while line_index < len(gt_lines) and match_beginning_of_line(gt_lines[line_index], index):
                x1, y1, x2, y2, im_class = get_data_from_line(gt_lines[line_index])
                # fx, fy = IM_WIDTH/oldw, IM_HEIGHT/oldh # account for resizing
                # x1, y1, x2, y2 = int(x1 * fx), int(y1 * fy), int(x2 * fx), int(y2 * fy)
                faces.append(img_to_array(array_to_img(img[y1:y2, x1:x2]).resize((FACE_WIDTH, FACE_HEIGHT))))
                # array_to_img(img[y1:y2, x1:x2]).resize((FACE_WIDTH, FACE_HEIGHT)).save(DATA_PATH + "faces/" + str(len(faces)) + ".jpg")
                faces_labeled.append((len(faces) - 1, im_class))
                # faces_coords.append((len(faces) - 1, x1, y1, x2, y2, FaceClasses.Face.value))
                valid_pixels[max(y1 - FACE_HEIGHT, 0):y2, max(x1 - FACE_WIDTH, 0):x2] = -1 # make sure we won't choose negative examples that contain faces
                line_index += 1

            # generate negative examples
            # i think they only make sense for facial detection, so i'm only going to generate negative examples for that
            for neg in range(NEG_EX_PER_IMG):
                vp = valid_pixels[valid_pixels > -1]

                if len(vp) == 0:
                    print(f"Only found {neg} negative examples for image {index}, from {name} dataset...")
                    break

                index = np.random.randint(0, len(vp), size=1)
                y, x = int(vp[index] / img.shape[0]), int(vp[index] % img.shape[0])
                bad_faces.append(img[y:y+FACE_HEIGHT, x:x+FACE_WIDTH])
                # array_to_img(img[y:y+FACE_HEIGHT, x:x+FACE_WIDTH]).save(DATA_PATH + "badfaces/" + str(index) + "_" + str(neg) + ".jpg")
                # bad_faces_coords.append((len(faces) - 1, x, y, x + FACE_WIDTH, y + FACE_HEIGHT))
                valid_pixels[y, x] = -1 # unlike with faces, overlapping negative examples shouldn't be too much of a problem,
                                        # but, at the same time, generating the same negative example over and over again isn't
                                        # probably the greatest idea

    faces = np.asarray(faces)
    # faces_coords = np.asarray(faces_coords)
    faces_labeled = np.asarray(faces_labeled)
    bad_faces = np.asarray(bad_faces)
    # bad_faces_coords = np.asarray(bad_faces_coords)
    return faces, faces_labeled, bad_faces





"""
This will generate both positive and negative examples. I'm using only one function since it should save some time
to only open the files once, only read 
"""
def get_examples():
    faces_npy = os.path.join(DATA_PATH, "faces.npy")
    if os.path.exists(faces_npy):
        faces = np.load(DATA_PATH + FACES + ".npy", allow_pickle=True)
        faces_labels = np.load(DATA_PATH + FACES_LAB + ".npy", allow_pickle=True)
        negatives = np.load(DATA_PATH + NEG + ".npy", allow_pickle=True)
    else:
        faces, faces_labels, negatives = load_raw_imgs()
        np.save(os.path.join(DATA_PATH, FACES + ".npy"), faces)
        np.save(os.path.join(DATA_PATH, FACES_LAB + ".npy"), faces_labels)
        np.save(os.path.join(DATA_PATH, NEG + ".npy"), negatives)
    return faces, faces_labels, negatives

def match_beginning_of_line_valid(line, filename):
    for index, char in enumerate(filename[27:]):
        if line[index] != filename[27 + index]:
            return False
    return True

def get_data_from_line_valid(line, filename):
    # it's assumed line begins with filename
    relevant_line = line[len(filename[27:]) + 1:].split(" ")
    im_class = relevant_line[4]
    if im_class[:4] == "bart":
        im_class = ImageClasses.Bart.value
    elif im_class[:5] == "homer":
        im_class = ImageClasses.Homer.value
    elif im_class[:5] == "marge":
        im_class = ImageClasses.Marge.value
    elif im_class[:4] == "lisa":
        im_class = ImageClasses.Lisa.value
    elif im_class[:7] == "unknown":
        im_class = ImageClasses.Unknown.value
    return int(relevant_line[0]), int(relevant_line[1]), int(relevant_line[2]), int(relevant_line[3]), im_class


def load_raw_valid():
    valid = []
    valid_labels = []
    imgs_path = os.path.join(VALID_PATH + VALID_IM_PATH + "/", "*.jpg")
    gt_path = os.path.join(VALID_PATH, VALID_IM_PATH + ".txt")
    gt_file = open(gt_path, "rt")
    gt_lines = gt_file.readlines()
    line_index = 0
    files = glob(imgs_path)
    for index, file in enumerate(files):
        img = Image.open(file)
        img = img_to_array(img)
        oldw, oldh = img.shape[1], img.shape[0]
        # img = img_to_array(array_to_img(img).resize((IM_WIDTH, IM_HEIGHT)))
        valid.append(img)
        while line_index < len(gt_lines) and match_beginning_of_line_valid(gt_lines[line_index], file):
            x1, y1, x2, y2, im_class = get_data_from_line_valid(gt_lines[line_index], file)
            # fx, fy = IM_WIDTH/oldw, IM_HEIGHT/oldh
            # x1, y1, x2, y2 = int(x1 * fx), int(y1 * fy), int(x2 * fx), int(y2 * fy)
            valid_labels.append((len(valid) - 1, x1, y1, x2, y2, im_class))
            line_index += 1
    # valid = np.asarray(valid, np.uint8)
    valid_labels = np.asarray(valid_labels, np.int32)
    return valid, valid_labels

def get_valid():
    valid_npy = os.path.join(DATA_PATH, "*.npy")
    if os.path.exists(valid_npy):
        valid = np.load(os.path.join(DATA_PATH, VALID + ".npy"), allow_pickle=True)
        valid_labels = np.load(os.path.join(DATA_PATH, VALID_LAB + ".npy"), allow_pickle=True)
    else:
        valid, valid_labels = load_raw_valid()
        np.save(os.path.join(DATA_PATH, VALID + ".npy"), valid)
        np.save(os.path.join(DATA_PATH, VALID_LAB + ".npy"), valid_labels)
    return valid, valid_labels