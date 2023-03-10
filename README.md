<h1 align="center">
  <br>
Black and white image colorization -  Deep Learning project
  <br>
  <img src="https://github.com/SalomonMalka/Black-and-white-image-colorization/blob/main/Resources/charlie.gif" height="200">
</h1>
  <p align="center">
    <a href="https://github.com/SalomonMalka">Salomon Malka</a> •
    <a href="https://github.com/aviv21">Aviv Ish Shalom</a>
  </p>


## Table of contents
- [Our final results](#Our-final-results)
- [Project goal and Motivation](#Project-goal-and-Motivation)
- [Repository Description](#repository-description)
- [How to use ?](#How-to-use-?)
- [Previous work](#Previous-work)
- [Architecture](#Architecture)
- [Loss functions](#Loss-functions)
- [Further development ideas](#further-development-ideas)
- [References](#References)


## Our final results
<p float="left">
  <img src="https://github.com/SalomonMalka/Black-and-white-image-colorization/blob/main/Resources/final_coloring_1.jpg" width="400" />
  <img src="https://github.com/SalomonMalka/Black-and-white-image-colorization/blob/main/Resources/final_coloring_2.jpg" width="400" /> 
  <img src="/img3.png" width="500" />
  <p align="center">

## Project goal and Motivation

The problem of colorizing an image is very interesting and was not possible to do automatically before machine learning was introduced.
One of the biggest challenges in this problem is finding a loss function that agree with human perception.

Our goal was Using different types of loss functions to find out which one works better for this task.
You can see the video presentation of the preject here: https://youtube.com/watch?v=kgDaKjT3Kwg&feature=shares 


## Repository Description

| Filename                    | description                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `Colorization_training.ipynb` | The main file in google colab format, including the loading of dataset and training. |
| `Colorization_testing.ipynb`    | A testing file in google colab format, including coloring some test set images and an option to color your own images.                                                                 |
| `Resources `                      | Folder consists of all the images from the project                                                |
| `trained unet using different loss functions`           | The pretrained weights for our final model trained using different loss functions                                          |
| `ProjectPresentation.pptx`           | The project presentation                                        |
    
<a name="How-to-use-?"/>

## How to use ?
To colorize your own image you have to open the `Colorization_testing.ipynb` file and load the parameters of the model (which appear in the `trained unet using different loss functions` file). You can upload your own image in colab and use the function Colorize() to colorize it. This is explained at the end of the file. The image has to be in format .jpg, .jpeg or .png.

## Previous work
We used an article about building Colorization algorithm as our base line for the project. 
Most of the sources we found use GAN architecture for their model, because it is capable of creating a loss function that estimate how “real” the image look, instead of more simple pixel to pixel loss functions. 



## Architecture
We used a Unet architecture for our model. For the downsampling part we used convolution blocks  with residual connection between each block. Inside each block we did batch normalization and used LeakyReLU as an activation function. The convolution is done using 4X4 kernels and 𝑠𝑡𝑟𝑖𝑑𝑒=2 (no pooling). For the upsampling we used ConvTranspose2d 𝑠𝑡𝑟𝑖𝑑𝑒=2 and batchnorm with ReLU activation.


## Loss functions
The most standard losses are MSE and L1 losses. However, when it comes to human perception of images, these loss functions are not very good. Because they compare pixel to pixel, they don’t see the image structure and thus as a loss function for this task we expected they will not work.
There are a few loss functions that compare images in a way that is closer to how human perceive images. One of them is called perceptual loss and the basic principle is to extract features from the predictions and ground truth using pretrained freezed image based network (we used vgg) and use MSE loss between those features instead of the actual images. 



## Further development ideas

1. Training with bigger dataset and on more epochs
2. Try different perceptual loss functions to see if they can give better results.
3. Use a pretrained semantic segmentation network and apply it before the input. This should considerably improve the results as the colorization is done with the semantic component information of the image

## References

- The article we used as a baseline for our project: [Colorizing black & white images with U-Net and conditional GAN — A Tutorial
](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)
- A paper that was very usefull to us: [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)
- Another paper that was usefull to us: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- Another article on the colorization problem: [Image Colorization with Convolutional Neural Networks](https://lukemelas.github.io/image-colorization.html)



