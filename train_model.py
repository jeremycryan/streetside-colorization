from PIL import Image
import numpy as np
import torch
#import torchvision
#import torchvision.transforms as transforms
import pandas
import os

color_dir = 'color_imgs'
grayscale_dir = 'grayscale_imgs'

def get_images(input_dir):
    paths = os.listdir(input_dir)
    img_paths = [item for item in paths if item.lower().endswith(".jpg")]
    for path in img_paths:
        if not path.endswith('_0.jpg') and not path.endswith('_5.jpg'):
            image = Image.open(os.path.join(input_dir, path))
            yield image

def format_color_images():
    imgs = list(get_images(color_dir))[:4]
    data = [np.expand_dims(np.array(i), axis=3) for i in imgs]
    data = np.concatenate((*data,), axis=3)
    return data

def format_grayscale_images():
    imgs = list(get_images(grayscale_dir))[:4]
    data = [np.expand_dims(np.array(i), axis=3) for i in imgs]
    data = np.concatenate((*data,), axis=2)
    return data

class ColorizeCNN(torch.nn.Module):
    def __init__(self):
        super(ColorizeCNN, self).__init__()

        

if __name__ == '__main__':
    data = format_grayscale_images()

    print(data.shape)