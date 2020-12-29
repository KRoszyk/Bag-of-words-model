import pickle
from pathlib import Path
from typing import Tuple
from sklearn import cluster
import cv2 as cv
import numpy as np
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import Augmentor


def make_augmentations():
    p = Augmentor.Pipeline("./../Projekt ZPO/train/")
    p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.3, percentage_area=0.8)
    p.flip_top_bottom(probability=0.1)
    p.sample(500)


def load_dataset(dataset_dir_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i, class_dir in enumerate(sorted(dataset_dir_path.iterdir())):
        for file in class_dir.iterdir():
            img_file = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
            x.append(img_file)
            y.append(i)

    return np.asarray(x), np.asarray(y)


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(
        data: np.ndarray,
        feature_detector_descriptor,
        vocab_model
) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors.astype(float), vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


def train_model(train_images, feature_detector_descriptor, model_name, NB_WORDS):
    train_descriptors = []

    for i, image in enumerate(train_images):
        print("Image ", i, "/", len(train_images) - 1)
        for descriptor in feature_detector_descriptor.detectAndCompute(image, None)[1]:
            train_descriptors.append(descriptor)

    kmeans = cluster.KMeans(n_clusters=NB_WORDS, random_state=42, verbose=True, algorithm='elkan')
    kmeans.fit(train_descriptors)

    pickle.dump(kmeans, open(model_name, 'wb'))


def train_classifier(train_images, train_labels, feature_detector_descriptor):
    with Path('model_SIFT128.p').open('rb') as vocab_file:
        vocab_model = pickle.load(vocab_file)

    X_train = apply_feature_transform(train_images, feature_detector_descriptor, vocab_model)
    y_train = train_labels

    classifier1 = svm.SVC()
    param_grid = {
        'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
        'C': [0.1, 0.5, 1, 3, 5, 10, 15, 20, 50, 100],
        'coef0': [0.0, 0.5, 1.0, 2.0, 3.0],
        'degree': [0, 3, 4, 5, 6, 7, 8],
        'gamma': [10, 100, 200, 300, 400]
    }

    grid_search = GridSearchCV(classifier1, param_grid)
    grid_search.fit(X_train, y_train)

    print(f'SVM train:{grid_search.score(X_train, y_train)}')

    pickle.dump(grid_search, open('clf_SIFT128.p', 'wb'))


def create_model():
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    start = time.time()
    np.random.seed(42)
    data_path = Path('./../Projekt ZPO/train/')  # You can change the path here to your train folder

    x, y = load_dataset(data_path)

    feature_detector_descriptor = cv.SIFT_create()

    train_model(x, feature_detector_descriptor, 'model_SIFT128.p', 128)
    train_classifier(x, y, feature_detector_descriptor)
    end = time.time()

    print(f'Time of learning the model: {time.strftime("%H:%M:%S", time.gmtime(end-start))}')


def test_model():
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    start = time.time()
    np.random.seed(42)
    data_path = Path('./../Projekt ZPO/test/')  # You can change the path here to your train folder

    x_test, y_test = load_dataset(data_path)

    feature_detector_descriptor = cv.SIFT_create()

    # TODO: train a vocabulary model and save it using pickle.dump function
    with Path('model_SIFT128.p').open('rb') as vocab_file:  # Don't change the path here
        vocab_model = pickle.load(vocab_file)

    x_transformed = apply_feature_transform(x_test, feature_detector_descriptor, vocab_model)

    # TODO: train a classifier and save it using pickle.dump function
    with Path('clf_SIFT128.p').open('rb') as classifier_file:  # Don't change the path here
        clf = pickle.load(classifier_file)

    score = clf.score(x_transformed, y_test)
    end = time.time()
    print(f'Result on testing images: {score}')

    print(f'Time of testing the model: {time.strftime("%H:%M:%S", time.gmtime(end-start))}')


if __name__ == '__main__':
    # train_model()             # uncomment to train your own model
    test_model()
