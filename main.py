from Code.IO.load_data import get_examples, get_valid
from Code.Model.cnn_bow import extract_features_facial_sift, train_svm_facial
from Code.Data_Processing.create_train_data import create_train_data_facial, normalize_train_data
from Code.Model.sliding_window import sliding_window_valid
from Code.Model.evaluate_results import evaluate_detections_facial

def main():
    faces, faces_coords, negatives = get_examples()
    valid, valid_labels = get_valid()
    s_pos_h, s_neg_h, s_cb = extract_features_facial_sift(faces, negatives)
    t_data, t_labels = create_train_data_facial(s_pos_h, s_neg_h)
    t_data = normalize_train_data(t_data)
    svm = train_svm_facial(t_data, t_labels)
    detections = sliding_window_valid(valid, s_cb, svm)
    evaluate_detections_facial(detections, valid_labels)
    # extract_features_facial_hog(faces, negatives)

if __name__ == "__main__":
    main()