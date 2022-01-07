import numpy as np
from Code.IO.load_data import NAMES
from Code.Logical.classes import ImageClasses

SOL_PATH = 'evaluare/fisiere_solutie/Tantaru_Dragos_Constantin_344/'

def write_detections_facial(detections, scores, file_names):
    np.save(SOL_PATH + 'task1/' + 'detections_all_faces.npy', detections)
    np.save(SOL_PATH + 'task1/' + 'scores_all_faces.npy', scores)
    np.save(SOL_PATH + 'task1/' + 'file_names_all_faces.npy', file_names)

def write_detections_classes(detections, scores, file_names):
    for name in NAMES:
        if name == 'bart':
            cr_name = ImageClasses.Bart.value
        elif name == 'homer':
            cr_name = ImageClasses.Homer.value
        elif name == 'lisa':
            cr_name = ImageClasses.Lisa.value
        elif name == 'marge':
            cr_name = ImageClasses.Marge.value
        elif name == 'unknown':
            cr_name = ImageClasses.Unknown.value
        cr_indexes = []
        for d_index, detection in enumerate(detections):
            *stuff, face, file_index = detection
            if face == cr_name:
                cr_indexes.append(d_index)
        cr_indexes = np.asarray(cr_indexes)

        if len(cr_indexes) > 0:
            det_relevant = detections[cr_indexes]
            scores_relevat = scores[cr_indexes]
            files_relevant = file_names[cr_indexes]
        else:
            det_relevant = np.asarray([])
            scores_relevat = np.asarray([])
            files_relevant = np.asarray([])
        np.save(SOL_PATH + 'task2/' + 'detections_' + name + '.npy', det_relevant)
        np.save(SOL_PATH + 'task2/' + 'scores_' + name + '.npy', scores_relevat)
        np.save(SOL_PATH + 'task2/' + 'file_names_' + name + '.npy', files_relevant)