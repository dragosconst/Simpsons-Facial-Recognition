from Code.IO.load_data import get_examples
from Code.Model.cnn_bow import extract_features_facial_sift

def main():
    faces, faces_coords, negatives = get_examples()
    s_pos_h, s_neg_h = extract_features_facial_sift(faces, negatives)
    # extract_features_facial_hog(faces, negatives)

if __name__ == "__main__":
    main()