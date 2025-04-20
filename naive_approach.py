import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import os

def generate_random_points(image, num_points):
    """
    It will take an image and place random points within the image size.
    """
    height, width = image.shape[:2]
    points = []
    for i in range(num_points):
        x = np.random.randint(0, width - 1)
        y = np.random.randint(0, height - 1)
        points.append((x, y))
    return points

def generate_heatmap_from_random_points(image, points):
    """
    This function will take the random points from the above and generate a Gaussian blob based heatmap 
    """
    height, width = image.shape[:2]
    density_map = np.zeros((height, width), dtype=np.float32)

    # For each point, create the Gaussian blob
    for (x, y) in points:
        blob = np.zeros((height, width), dtype=np.float32)
        cv2.circle(blob, (x, y), radius=10, color=1.0, thickness=-1)
        blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=50, sigmaY=50)
        density_map += blob

    return density_map

def heatmap_comparison(image_folder, heatmap_folder, points):
    """
    This function will iterate through each of the test images and heatmaps and compare the randomly created one with the ground truth one
    """

    mse_list = []
    
    #Iterate through each image in the folder
    for image in tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image)
        heatmap_path = os.path.join(heatmap_folder, f"heatmap_{image}")

        #Load the image and heatmap
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        #create the random points and the corresponding heatmap
        random_points = generate_random_points(image, points)
        random_heatmap = generate_heatmap_from_random_points(image, random_points)
        random_heatmap_normalized = cv2.normalize(random_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        random_heatmap_normalized = np.uint8(random_heatmap_normalized)

        #Calculate the MSE between the two heatmaps
        mse = mean_squared_error(heatmap.flatten(), random_heatmap.flatten())
        mse_list.append(mse)

    # Print the average MSE
    average_mse = np.mean(mse_list)
    print(f"Average MSE: {average_mse}")

if __name__ == "__main__":
    # Get the test image and heatmap folders
    image_folder = "Dataset/test/images"
    heatmap_folder = "Dataset/test/heatmaps"
    
    # Number of random points. I kept it at 30 because my ground truth had 30 individuals
    num_points = 30

    # Run the heatmap comparison
    heatmap_comparison(image_folder, heatmap_folder, num_points)