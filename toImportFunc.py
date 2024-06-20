import pickle
import numpy as np
import pandas as pd
import os
from PIL import Image
import random

def load_class_ids(file_path):
    # Load class IDs from a pickle file
    with open(file_path, 'rb') as f:
        class_ids = pickle.load(f, encoding='latin1')
    return class_ids

def load_embeddings_data(file_path):
    # Load embeddings from a pickle file and convert to numpy array
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f, encoding='latin1')
    embeddings_array = np.array(embeddings)
    print('Embeddings shape:', embeddings_array.shape)
    return embeddings_array

def load_filenames_list(file_path):
    # Load filenames from a pickle file
    with open(file_path, 'rb') as f:
        filenames = pickle.load(f, encoding='latin1')
    return filenames

def load_bboxes(dataset_directory):
    # Load bounding boxes and image file names from the dataset directory
    bounding_boxes_file = os.path.join(dataset_directory, 'bounding_boxes.txt')
    image_files_file = os.path.join(dataset_directory, 'images.txt')

    bbox_df = pd.read_csv(bounding_boxes_file, sep='\s+', header=None).astype(int)
    file_names_df = pd.read_csv(image_files_file, sep='\s+', header=None)

    # Extract list of file names
    file_names = file_names_df[1].tolist()

    # Create a dictionary to store bounding boxes by file name
    bbox_dict = {img_file[:-4]: [] for img_file in file_names[:2]}

    for i in range(len(file_names)):
        # Assign each bounding box to its corresponding image file
        bbox = bbox_df.iloc[i][1:].tolist()
        key = file_names[i][:-4]
        bbox_dict[key] = bbox

    return bbox_dict

def load_image(image_path, bbox, target_size):
    # Open and process an image using its bounding box
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    if bbox is not None:
        R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - R)
        y2 = np.minimum(height, center_y + R)
        x1 = np.maximum(0, center_x - R)
        x2 = np.minimum(width, center_x + R)
        image = image.crop([x1, y1, x2, y2])
    image = image.resize(target_size, Image.BILINEAR)
    return image

def load_data(filenames_path, class_info_path, dataset_directory, embeddings_path, img_size):
    # Load and prepare the dataset
    filenames = load_filenames_list(filenames_path)
    class_ids = load_class_ids(class_info_path)
    bounding_boxes = load_bboxes(dataset_directory)
    all_embeddings = load_embeddings_data(embeddings_path)

    images, labels, embeddings = [], [], []

    print("All embeddings shape:", all_embeddings.shape)

    for index in range(min(len(filenames), len(all_embeddings))):
        filename = filenames[index]
        bbox = bounding_boxes[filename]

        try:
            # Load and process each image
            image_file = '{}/images/{}.jpg'.format(dataset_directory, filename)
            image = load_image(image_file, bbox, img_size)

            embedding_list = all_embeddings[index, :, :]

            random_embedding_index = random.randint(0, embedding_list.shape[0] - 1)
            selected_embedding = embedding_list[random_embedding_index, :]

            images.append(np.array(image))
            labels.append(class_ids[index])
            embeddings.append(selected_embedding)
        except Exception as e:
            print(e)

    images_array = np.array(images)
    labels_array = np.array(labels)
    embeddings_array = np.array(embeddings)
    return images_array, labels_array, embeddings_array
