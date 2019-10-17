from PIL import Image
import numpy as np
import torch
import torch.utils.data as d
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
#import torchvision
#import torchvision.transforms as transforms
from skimage.color import lab2rgb, rgb2lab
import pandas
import os
import random
import time

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
    imgs = list(get_images(color_dir))
    data = [np.expand_dims(np.array(i), axis=3) for i in imgs]
    data = np.concatenate((*data,), axis=3)
    return data

def format_grayscale_images():
    imgs = list(get_images(grayscale_dir))
    data = [np.expand_dims(np.array(i), axis=3) for i in imgs]
    data = np.asarray((*data,))
    return data

class ColorizeCNN(nn.Module):
    def __init__(self):
        super(ColorizeCNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)

        kernel_size = 3
        pad_size = kernel_size // 2
        self.conv1 = nn.Conv2d(1, 8, kernel_size, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size, padding=0)
        self.conv5 = nn.Conv2d(64, 2, kernel_size, padding=0)
        self.pad = lambda x: functional.pad(x, (pad_size, pad_size, pad_size, pad_size), mode='replicate')

    def forward(self, x):

        x = self.pool(functional.relu(self.conv1(self.pad(x))))
        x = self.pool(functional.relu(self.conv2(self.pad(x))))
        
        x = torch.sigmoid(self.conv3(self.pad(x)))
        x = torch.sigmoid(self.conv4(self.pad(self.upsample(x))))
        x = torch.sigmoid(self.conv5(self.pad(self.upsample(x))))

        return x
    

if __name__ == '__main__':
    net = ColorizeCNN()

    print('Starting')

    imgs = get_images(color_dir)
    labels = []
    for img in imgs:
        img_lab = rgb2lab(np.asarray(img))
        img_lab = (img_lab + 128) / 255
        img_ab = img_lab[:, :, 1:3]
        img_ab = img_ab.transpose((2, 0, 1))
        labels.append(img_ab)

    print('Loaded labels')

    grayscale_data = format_grayscale_images()
    training_data = []
    for img in grayscale_data:
        img = img.transpose((2, 0, 1))
        img = np.asarray(img)
        training_data.append(img)

    print('Loaded training data')

    # define loss function
    criterion = nn.MSELoss()

    # define optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.0)

    print('Starting Lookup List Creation')

    lookup = list(range(len(labels)))
    random.shuffle(lookup)

    training_data = [training_data[lookup[i]] for i in range(len(training_data))]
    training_data = torch.from_numpy(np.asarray(training_data)).float()
    labels = [labels[lookup[i]] for i in range(len(labels))]
    labels = torch.from_numpy(np.asarray(labels)).float()

    print('Starting training')

    start_time = time.time()

    running_loss = 0
    batch_size = 16
    losses = [0]*10
    test_size = 10
    # forward + backward + optimize
    #for i in range(0, len(lookup) - (test_size + 1)*batch_size, batch_size):
    for i in range(0, 192, batch_size):
        data = training_data[i:i+batch_size]

        # zero the parameter gradients
        optimizer.zero_grad()

        output = net(data)
        loss = criterion(output, labels[i:i+batch_size])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % batch_size == 0:
            losses.insert(0, running_loss)
            losses.pop()
            print(f'Idx: {i}, Running l ll ll L: {sum(losses)/len(losses)}')
            running_loss = 0

    total_time = time.time() - start_time

    print(total_time)

    for i in range(batch_size*test_size):
        image_array = training_data[-i].numpy() * 100/255

        output = net(training_data[-batch_size*test_size:])
        output = output.data.numpy()

        img = output[-i]
        img = img - 0.5
        intensity = 50
        img[0] = img[0] * intensity / np.max(np.abs(img[0]))
        img[1] = img[1] * intensity / np.max(np.abs(img[1]))
        img = np.concatenate((image_array, img), axis=0)
        img = img.transpose(1, 2, 0)

        img = lab2rgb(img)

        img = Image.fromarray(np.uint8(img * 255))

        img.save("some-ort_of-image" + str(i) + ".jpg", "JPEG")


    print('Done')


# l: 0 - 100
# a: -86.185 - 98.254
# b: -107.863 - 94.482