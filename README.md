# Streetside Colorization
### Jeremy Ryan and Nate Sampo

Project Code: https://github.com/jeremycryan/streetside-colorization

![Demo image](https://github.com/jeremycryan/streetside-colorization/blob/master/sample.jpg?raw=true)


## Overview
One of the largest issues with grayscale images is their lack of color. To that end, we aimed to develop a convolutional machine learning model that would add color to colorless Google Streetview images. Utilizing the popular machine learning library PyTorch, we successfully brought vibrant color back to monotonously monotone Streetview images.


## Dataset
Our dataset came from the Center for Research in Computer Vision (https://www.crcv.ucf.edu/data/GMCP_Geolocalization/), and was collected for use in a paper on the application of computer vision in geo-localization.<sup>1</sup> This dataset provides 62,058 images of urban areas within the United States, with each image being of size 1280x1024. These images typically contained normal urban features such as the sky, clouds, buildings, roads, trees, cars, and people. Pictured below is a relatively standard image from the dataset.

![Example image from the dataset](https://i.imgur.com/zWy3xPE.jpg)

*Example image from the dataset*


## Preprocessing
We immediately noticed some small issues with the dataset that would make this dataset difficult to use for our purposes. Perhaps most glaringly, most of the images include some vestigial Google Streetview GUI elements. Additionally, they are much too large to train a convolutional network on with our limited computational and time resources. Luckily, the GUI elements can be easily cropped out, and the image can be downscaled to a more reasonable size. After cropping and downscaling the images by a factor of 11 to size 112x88, we had data that we were confident would work well for us.

The images were then converted from RGB color space to LAB color space --- Lightness, Alpha, Beta --- to create both our training data and our labels. Our input training data is simply the Lightness channel of the image, effectively a grayscale representation devoid of color, while the labels are the other two channels, as these channels represent the colors that the model should output. After passing the Lightness channel input data through the network, we can append on the other two trained output channels and convert the image back into RGB color space to be displayed. An example grayscale (training) image and color (label) image are pictured below.


![Grayscale Image (training)](https://i.imgur.com/26dwF7p.jpg) ![Color Image (label)](https://i.imgur.com/dt7WgkR.jpg)

*Cropped and Downscaled Images*


## Model

Below is a block diagram of our final network.


![Network block diagram](https://github.com/jeremycryan/streetside-colorization/blob/master/block_diagram.png?raw=true)

*Convolutional Network Block Diagram*

In order to give our network a high degree of agency over the output, we put our data through a series of several convolutions and sampling operations. 

The outermost colvolutions should identify highly local features in a 3px by 3px frame --- for instance, noisy textures like tree leaves --- while the innermost convolutions should identify larger features, drawing influence from more pixels in the original image.

Here is our PyTorch implementation:

```
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
        self.pad = lambda x: functional.pad(x, 
            (pad_size, pad_size, pad_size, pad_size),
            mode='replicate')

    def forward(self, x):

        x = self.pool(functional.relu(self.conv1(self.pad(x))))
        x = self.pool(functional.relu(self.conv2(self.pad(x))))
        
        x = torch.sigmoid(self.conv3(self.pad(x)))
        x = torch.sigmoid(self.conv4(self.pad(self.upsample(x))))
        x = torch.sigmoid(self.conv5(self.pad(self.upsample(x))))

        return x
```

There are a few parts of our implementation that are worth noting.

#### Dimensionality and Padding

Because our network is using images of the same size as inputs and outputs, we have to be careful about maintaining the same dimensionality as we process data. As such, we pad the data with one pixel each time we perform a convolution, thus maintaining the same size. In order to minimize the impact on training near the edges of the image, we used a 'replicate' padding mode, rather than the default padding on ```Conv2D```, which adds zeros.

#### Activation Functions

During the downsampling of the network, we use ```functional.relu``` as an activation function. During upsampling, we instead use ```torch.sigmoid```. Prior, we had used only ReLU, but found that our outputs tended toward extreme values. Our reasoning was that a sigmoid function would prevent extreme values from overpowering other values in the image through the convolution and upsampling in the latter part of the network.

#### Cost Function

Our network calculates loss using mean squared error (```nn.MSELoss```). Although this was our go-to cost function, it may cause our results to tend toward more neutral values because large errors are punished so highly.

## Testing and Validation

We trained our network on 6,000 of our sample images, with a batch size of 16. We calculated the loss for each batch and plotted it over time, with a moving average of ten batches.


![Training Loss Over Time](https://github.com/jeremycryan/streetside-colorization/blob/master/training_loss.png?raw=true)

*Mean Squared Error Loss vs Batch Number*


In the first batches, up to batch 60 or so, loss decreases approximately exponentially, reaching a minimum of just under 7 x 10<sup>-4</sup>. After this point, it stays approximately steady, even though we continue training for a few hundred batches.

This suggests that we aren't optimally using our data. It's possible this loss could be further reduced with larger batch sizes, a slower rate of learning, or a deeper network.

Additionally, our outputs tended to be very close to monochrome (with alpha and beta values close to zero). We think that the mean squared error loss function could be contributing to this behavior, since the network would avoid guessing extreme values unless it had a very high degree of confidence.

The following images are example outputs from our network, with normalized AB values:


![](https://i.imgur.com/A2KQThc.jpg)![](https://i.imgur.com/hnvzNhc.jpg) ![](https://i.imgur.com/XyTb0pX.jpg)![](https://i.imgur.com/6QX3LHQ.jpg)

![](https://i.imgur.com/wKgck61.jpg)![](https://i.imgur.com/gOYnfDM.jpg) ![](https://i.imgur.com/UfU6ocW.jpg)![](https://i.imgur.com/nclldcy.jpg)

![](https://i.imgur.com/oU5Wqle.jpg)![](https://i.imgur.com/WVV08me.jpg) ![](https://i.imgur.com/DcgcViy.jpg)![](https://i.imgur.com/ifzM7wD.jpg)

![](https://i.imgur.com/wSPU8os.jpg)![](https://i.imgur.com/QvcAO1Q.jpg) ![](https://i.imgur.com/0muSWbM.jpg)![](https://i.imgur.com/VWgVCWU.jpg)

*Grayscale images compared to our colorized versions*



## Explorations

We performed a few additional experiments to see their effects on the neural network.

#### L1 vs MSE Cost Function

We experimented with using ```nn.L1Loss``` instead of ```nn.MSELoss``` to train our network. ```nn.L1Loss``` is a Mean Absolute Error loss function which simply returns the average of the absolute values of the differences between the expected values and the output values, while ```nn.MSELoss``` is a Mean Squared Error loss function that returns the mean of the squares of the differences.

The following is a plot of loss over time with each of these training methods:


![Plot of L1 Loss vs MSE Loss](https://github.com/jeremycryan/streetside-colorization/blob/master/training_loss_l1.png?raw=true)

*MSE Loss & L1 Loss vs Batch Number*


The mean squared error function appears to reach a minimum and stabilize significantly faster than the L1 loss function with the same learning rate, reaching its minimum around batch 50 compared to approximately batch 100. 

#### Kernel Size and Timing

We experimented with the timing of different kernel sizes when training our network.


![Graph of Training Time by Kernel Size](https://github.com/jeremycryan/streetside-colorization/blob/master/training_time_kernel.png?raw=true)

*Training Time vs Kernel Width*


The training time of the network appears to increase exponentially with respect to kernel size for values in the range [1, 7]. The training time for a 9x9 kernel was significantly higher, but we suspect that the bottleneck was actually the computer's memory management rather than compute time.

This suggests that it is likely more efficient to use multiple 3x3 kernels in a network, a larger kernel with equivalent reach --- they can accomplish more complex logic, with as much positional reach, in comparable or lesser time.

## Future Work
While this model works well for urban environments and is able to detect many of the urban features, it is likely not scalable to other kinds of landscapes as the source of all training data comes from cities. This could be significantly improved with a more varied dataset and additional complexity within the model to aid in picking up on a more diverse set of features.

Additionally, the input and output images are quite small. With additional time and resources, this model could be trained on larger images to provide a potentially more useful and interesting output.

Overall we are quite proud of what we were able to accomplish in this project. We successfully trained a convolutional machine learning model to detect and somewhat accurately color features in grayscale urban environments.


&nbsp;
&nbsp;
Footnote 1: Amir Roshan Zamir and Mubarak Shah, *"Image Geo-localization Based on Multiple Nearest Neighbor Feature Matching using Generalized Graphs"*, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2014
