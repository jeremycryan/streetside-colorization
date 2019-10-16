from PIL import Image
import numpy as np
import torch
import torch.utils.data as d
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
from skimage.color import lab2rgb, rgb2lab
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
    imgs = list(get_images(color_dir))
    data = [np.expand_dims(np.array(i), axis=3) for i in imgs]
    data = np.concatenate((*data,), axis=3)
    return data

def format_grayscale_images():
    imgs = list(get_images(grayscale_dir))
    data = [np.expand_dims(np.array(i), axis=3) for i in imgs]
    data = np.asarray((*data,))
    print(data.shape)
    #data = np.concatenate((*data,), axis=2)
    return data

class ColorizeCNN(nn.Module):
    def __init__(self):
        super(ColorizeCNN, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv3a = nn.Conv2d(4, 4, 3, padding=1)
        self.conv3b = nn.Conv2d(4, 4, 3, padding=1)
        self.conv4a = nn.Conv2d(4, 2, 3, padding=1)
        self.conv4b = nn.Conv2d(4, 2, 3, padding=1)
        self.conv5a = nn.Conv2d(2, 1, 3, padding=1)
        self.conv5b = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        y = x
        x = F.relu(self.conv3a(x))
        y = F.relu(self.conv3b(y))
        x = F.relu(self.conv4a(self.upsample(x)))
        y = F.relu(self.conv4b(self.upsample(y)))
        x = F.relu(self.conv5a(self.upsample(x)))
        y = F.relu(self.conv5b(self.upsample(y)))

        #print(type(x))
        return torch.cat([x, y], 1)
    

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

    labels = torch.from_numpy(np.asarray(labels)).float()

    print('Loaded labels')

    grayscale_data = format_grayscale_images()
    training_data = []
    for img in grayscale_data:
        img = img.transpose((2, 0, 1))
        img = np.asarray(img)
        training_data.append(img)

    training_data = torch.from_numpy(np.asarray(training_data)).float()

    print('Loaded training data')

    # define loss function
    criterion = nn.MSELoss()

    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # zero the parameter gradients
    optimizer.zero_grad()

    print('Starting training')

    running_loss = 0
    batch_size = 8
    # forward + backward + optimize
    for i in range(0, len(training_data) - batch_size*2, batch_size):
    #for i in range(0, batch_size, batch_size):
        data = training_data[i:i+batch_size]

        output = net(data)
        loss = criterion(output, labels[i:i+batch_size])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            print(f'Idx: {i}, Running l ll ll L: {running_loss}')
            running_loss = 0

    image_array = training_data[-1].numpy()

    output = net(training_data[-batch_size:])
    output = output.data.numpy()

    img = output[-1]
    img = np.concatenate((image_array, img), axis=0)
    img = img.transpose(1, 2, 0)
    print(img)
    img = lab2rgb(img)
    img = Image.fromarray(np.uint8(img))
    img.save("some-ort_of-image.jpg", "JPEG")


    print('Done')