import numpy as np
import cv2
import os
import json
import shutil

def generate_density_map(json_path):
    """
    This function will take a json path of characters and their screen locations. It will use Gaussian Blur to make a heatmap, where the closer some points are, the more 'warm' the color is.
    """

    # Some set variables
    height = 1080
    width = 1920
    #Padding for blur to take care of edge cases
    padding = 100
    padded_height = height + padding
    padded_width = width + padding

    #Create an empty density map
    density_map = np.zeros((padded_height, padded_width), dtype=np.float32)

    #Opent the json file
    with open(json_path, 'r') as f:
        data = json.load(f)

    #Go through every character and get their position
    for character in data['characters']:
        x = int(character['screenPosition']['x']) + padding 
        y = int(character['screenPosition']['y']) + padding

        # We then add a Gaussian blur
        if 0 < x < padded_width and 0 < y < padded_height:
            blob = np.zeros((padded_height, padded_width), dtype=np.float32)
            cv2.circle(blob, (x, y), radius = 10, color = 1.0, thickness=-1)
            blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=50, sigmaY=50)
            density_map += blob

    #Once we are done, we want to remove the padding
    density_map = density_map[padding:padding + height, padding:padding + width]
    return density_map

def generate_heatmaps(dataset_folder):
    """
    This function will take a dataset folder and iterate through each one, using generate_density_maps to have for each image.
    This may take a while, so be patient.
    """

    # Get the folders
    image_folder = os.path.join(dataset_folder, 'crowd_images')
    json_folder = os.path.join(dataset_folder, 'crowd_jsons')
    
    heatmap_folder = os.path.join(dataset_folder, 'crowd_heatmaps')
    os.makedirs(heatmap_folder, exist_ok=True)

    #Iterate through each image in the dolder
    for image_file in os.listdir(image_folder):
        #Get the correspoinding json file
        # Can be either pngs or jpgs
        json_file = image_file.replace('.png', '.json').replace('.jpg', '.json')

        #Get the full path
        image_path = os.path.join(image_folder, image_file)
        json_path = os.path.join(json_folder, json_file)

        # Make sure that both exist
        # If yes, generate the heatmap
        if os.path.exists(image_path) and os.path.exists(json_path):
            print(f"Generating heatmap for {image_file}...")
            
            density_map = generate_density_map(json_path)

            # Normalize the density map
            density_map_normalized = cv2.normalize(density_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            density_map_normalized = np.uint8(density_map_normalized)

            # We then save the heatmap image
            heatmap_filename = f"heatmap_{os.path.splitext(image_file)[0]}.png"
            heatmap_path = os.path.join(heatmap_folder, heatmap_filename)
            cv2.imwrite(heatmap_path, density_map_normalized)
            print(f"Heatmap saved to {heatmap_path}")
        else:
            print(f"Skipping {image_file} as the corresponding json file does not exist.")


def train_val_test_split():
    """
    This function will split the dataset into train, validation and test sets.
    It will create and store them within three folders
    """

    #Set the random seed for reproducibility
    np.random.seed(0)

    image_folder = 'Dataset/crowd_images'
    heatmap_folder = 'Dataset/crowd_heatmaps'

    # Sort and shuffle the files
    images = sorted(os.listdir(image_folder))
    np.random.shuffle(images)

    #Make the split
    total = len(images)
    train_split = int(0.7 * total)
    val_split = int(0.85 * total)

    # Creates a ductionary for splits
    splits = {
        'train': images[:train_split],
        'val': images[train_split:val_split],
        'test': images[val_split:]
    }

    #Iterates through each split and sort them into their respective folders
    for split in splits:
        os.makedirs(f'Dataset/{split}/images', exist_ok=True)
        os.makedirs(f'Dataset/{split}/heatmaps', exist_ok=True)

        for image in splits[split]:
            base = os.path.splitext(image)[0]
            heatmap = f"heatmap_{base}.png"

            shutil.copy(os.path.join(image_folder, image), f'Dataset/{split}/images/{image}')
            shutil.copy(os.path.join(heatmap_folder, heatmap), f'Dataset/{split}/heatmaps/{heatmap}')

    print("Dataset split into train, val and test sets.")

if __name__ == "__main__":
    dataset_folder = 'Dataset'
    generate_heatmaps(dataset_folder)
    print("Heatmaps generated.")
    train_val_test_split()
    print("Train, val and test sets created.")