import os
import numpy as np
from glob import glob
import cv2 as cv
from Code.Logical.classes import ImageClasses, FaceClasses

TRAIN_PATH = "antrenare/"
DATA_PATH = "data/"
NAMES = ["bart", "homer", "lisa", "marge"]
IMGS_PER_CHAR = 1101  # holds true for all names
CHARS = 4
IMGS = CHARS * IMGS_PER_CHAR
IM_WIDTH = 64
IM_HEIGHT = 128
FACE_WIDTH = 16
FACE_HEIGHT = 16
NEG_EX_PER_IMG = 10

"""
Will load all data in a gigantic array of images. Labels are loaded in a (x1, y1, x2, y2, c) tuple, where c is the class
type
"""

# check if this line belongs to the given pic
def match_beginning_of_line(line, pic_index):
    STARTING_TEXT_LEN = len("pic_")
    for index, char in line:
        if index < STARTING_TEXT_LEN:
            continue
        number = int(line[index:index + 4])
        if number == pic_index:
            return True
        return False

def get_data_from_line(line):
    SKIP_CHARS = len("pic_0000.jpg")
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
    faces_coords_labeled = []
    faces = []
    faces_coords = []
    for name in NAMES:
        imgs_path = os.path.join(TRAIN_PATH + name + "/", "*.jpg")
        gt_path = os.path.join(TRAIN_PATH + name, '.txt')
        gt_file = open(gt_path, "rt")
        gt_lines = gt_file.readlines()
        line_index = 0
        files = glob(imgs_path)
        for index, file in enumerate(files):
            img = cv.imread(file)
            oldw, oldh = img.shape[0], img.shape[1]
            img = cv.resize(img, (IM_WIDTH, IM_HEIGHT))
            # generate positive examples
            pos_coords = []
            valid_pixels = [[i * IM_HEIGHT + j for j in range(IM_WIDTH)] for i in range(IM_HEIGHT)]
            valid_pixels = np.array(valid_pixels)
            while match_beginning_of_line(gt_lines[line_index], index):
                x1, y1, x2, y2, im_class = get_data_from_line(gt_lines[line_index])
                fx, fy = IM_WIDTH/oldw, IM_HEIGHT/oldh # account for resizing
                x1, y1, x2, y2 = int(x1 * fx), int(y1 * fy), int(x2 * fx), int(y2 * fy)
                faces.append(cv.resize(img[y1:y2, x1:x2],(FACE_WIDTH, FACE_HEIGHT)))
                faces_coords_labeled.append((x1, y1, x2, y2, im_class))
                faces_coords.append((x1, y1, x2, y2, FaceClasses.Face.value))
                valid_pixels[y1:y2, x1:x2] = -1
                pos_coords.append((x1, y1, x2, y2))
                line_index += 1

            # generate negative examples
            # first idea: randomly choose segments that are outside the bounds of any faces
            # this could be quite slow, due to having to compute all valid pixels for each image
            for neg in range(NEG_EX_PER_IMG):
                vp = valid_pixels[valid_pixels > -1]





"""
This will generate both positive and negative examples. I'm using only one function since it should save some time
to only open the files once, only read 
"""
def generate_examples():
    for name in NAMES:
        imgs_path = os.path.join(TRAIN_PATH + name + "/", "*.jpg")
        files = glob(imgs_path)
        for index, file in enumerate(files):
            img = cv.imread(file)

