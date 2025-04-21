import cv2
import numpy as np
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import joblib

def extract_gabor_features(image, window_size):
    """
    Extract Gabor features from the given window of the image
    """
    kernels = [
        cv2.getGaborKernel((4,4), 5.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F),
        cv2.getGaborKernel((4,4), 5.0, np.pi/2, 10.0, 0.5, 0, ktype=cv2.CV_32F),
        cv2.getGaborKernel((4,4), 5.0, np.pi/3, 10.0, 0.5, 0, ktype=cv2.CV_32F),
    ]

    gabor_features = []
    for kernel in kernels:
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        gabor_features.append(filtered.flatten())

    gabor_image = np.sum(gabor_features, axis=0)

    #Return a flattened variant to reduce the dimensionality
    return gabor_image.flatten()

def sliding_window(image, heatmap, window_size):
    """
    This function will take an image and a heatmap and slide a preset window across it
    This window size would be set by the user, where larger windows lead to faster data loading
    For each window of the image, the Gabor features will be extracted
    For each window of the heatmap, the largest value will be extracted and set as our y value
    """
    #Sotres features and labels
    X = []
    y = []

    #Get the dimensions for the heatmap and image
    height, width = image.shape

    # Loop through the image and heatmap
    for y_pos in range(0, height - window_size[0], window_size[0]):
        for x_pos in range(0, width - window_size[1], window_size[1]):
            # Get the window of the image
            image_window = image[y_pos:y_pos + window_size[0], x_pos:x_pos + window_size[1]]

            # Get the window of the heatmap
            heatmap_window = heatmap[y_pos:y_pos + window_size[0], x_pos:x_pos + window_size[1]]

            # Extract Gabor features from the image window
            features = extract_gabor_features(image_window, window_size)

            # Get the maximum value from the heatmap window as the label
            label = np.max(heatmap_window)

            X.append(features)
            y.append(label)
    
    #Conver to the numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y

def load_and_extract(image_folder, heatmap_folder, window_size):
    """
    This function will iterate through each image and heatmap in the folder and extract the Gabor features and labels
    """
    X = []
    y = []

    # Get the image and file heatmap names
    image_files = sorted(os.listdir(image_folder))
    heatmap_files = sorted(os.listdir(heatmap_folder))

    for image_file, heatmap_file in tqdm(zip(image_files, heatmap_files), total=len(image_files)):
        # Load the image and heatmap
        image_path = os.path.join(image_folder, image_file)
        heatmap_path = os.path.join(heatmap_folder, heatmap_file)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)

        # Extract features and labels using sliding window
        features, labels = sliding_window(image, heatmap, window_size)

        X.append(features)
        y.append(labels)
    
    #Flatten the lists
    X = np.vstack(X)
    y = np.concatenate(y)

    return X, y

def train_and_test_sgd(X_train, y_train, X_test, y_test):
    """
    This function will train the SGD model on the training data
    """
    #scale the X_train data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.pkl')

    # Transform the features to introduce non-linearity
    rbf_feature = RBFSampler(gamma=0.1, random_state=0)
    X_train_rbf = rbf_feature.fit_transform(X_train_scaled)
    X_test_rbf = rbf_feature.transform(X_test_scaled)

    # Save the RBF feature transformer for later use
    joblib.dump(rbf_feature, 'models/rbf_feature.pkl')

    # Train the SGD Regressor
    sgd = SGDRegressor(max_iter=1000, tol=0.001, verbose=1)
    sgd.fit(X_train_rbf, y_train)

    print("Training completed.")

    # Save the model
    joblib.dump(sgd, 'models/sgd_rbf_model.pkl')

    # Predict on the test set
    y_pred = sgd.predict(X_test_rbf)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    


if __name__ == "__main__":
    train_image_folder = 'Dataset/train/images'
    train_heatmap_folder = 'Dataset/train/heatmaps'
    test_image_folder = 'Dataset/test/images'
    test_heatmap_folder = 'Dataset/test/heatmaps'

    window_size = (16,16)
    X_train, y_train = load_and_extract(train_image_folder, train_heatmap_folder, window_size)
    X_test, y_test = load_and_extract(test_image_folder, test_heatmap_folder, window_size)

    train_and_test_sgd(X_train, y_train, X_test, y_test)
    print("Training and testing completed.")

