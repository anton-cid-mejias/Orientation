from PIL import Image
import glob
import numpy as np
import pandas as pd
from src.coco_data import resize_image
from tqdm import tqdm

def get_mask(img):
    return np.uint8((img > 0) * 255)

# Return a tuple with the bottom left corner of the bb and its width and height
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    xmin = cmin
    xmax = cmax
    ymin = rmin
    ymax = rmax

    return xmin, xmax, ymin, ymax

def crop_figure(img):
    mask = get_mask(img)
    xmin, xmax, ymin, ymax = bbox(mask)
    cropped_img = img[ymin:ymax,xmin:xmax]
    return cropped_img

def save_annotation(data, filename):
    data.to_csv(filename)

def load_data(file, images):
    image_list = []
    angles = pd.read_csv(file, delimiter=',', header=None)

    for filename in glob.glob(images + "/*png"):
        image_list.append(filename)

    angles['image_name'] = image_list
    angles.columns = ['w', 'x', 'y', 'z', 'image_name']
    return angles

def load_dataset(file, images, size):
    dataset = load_data(file, images)
    train = dataset.sample(frac=0.8).reset_index()
    val = dataset.drop(train.index).reset_index()

    train_images = np.zeros((train.shape[0], size[0], size[1], 3))
    train_angles = np.zeros((train.shape[0], 4))

    val_images = np.zeros((val.shape[0], size[0], size[1], 3))
    val_angles = np.zeros((val.shape[0], 4))

    # Load train
    print("Loading training dataset...")
    for index, row in tqdm(train.iterrows(), total=train.shape[0]):
        filename = row['image_name']
        train_angles[index, :] = np.array([row['w'], row['x'], row['y'], row['z']])

        im = np.array(Image.open(filename))
        im = crop_figure(im)
        im, _, _, _, _ = resize_image(im, min_dim=size[0], max_dim=size[1])
        train_images[index, :, :, :] = im[:, :, 0:3]

    # Load val
    print("Loading validation dataset...")
    for index, row in tqdm(val.iterrows(), total=val.shape[0]):
        filename = row['image_name']
        val_angles[index, :] = np.array([row['w'], row['x'], row['y'], row['z']])

        im = np.array(Image.open(filename))
        im = crop_figure(im)
        im, _, _, _, _ = resize_image(im, min_dim=size[0], max_dim=size[1])
        val_images[index, :, :, :] = im[:, :, 0:3]

    return train_images, train_angles, val_images, val_angles
