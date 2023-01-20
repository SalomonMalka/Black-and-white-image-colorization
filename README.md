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
- [Previous work](#Previous-work)
- [Architecture](#Architecture)
- [Loss functions](#Loss-functions)
- [Further development ideas](#further-development-ideas)
- [References](#References)
- [Notes](#Notes)


## Our final results


## Project goal and Motivation

The problem of colorizing an image is very interesting and was not possible to do in automatically before machine learning was introduced.
One of the biggest challenges in this problem is finding a loss function that

Our goal was Using different types of loss functions to find out which one works better for this task.


## Repository Description

| Filename                    | description                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `final_model_training.ipynb` | The main file in google colab format, including the loading of dataset and training. to open import to google colab |
| `test set and demo.ipynb`    | A testing file in google colab format, including coloring some test set images and an option to color your own images. to open import to google colab                                                                    |
| `resorces `                      | Folder consists of all the images from the project                                                |
| `trained unet using different loss functions`           | the pretrained whights for our final model trained using different loss functions                                          |
| `FinalProjectDeepLearning.pptx`           | The project presentation                                        |


## Previous work
We used an article about building Colorization algorithm as our base line for the project. 
Most of the sources we found use GAN architecture for their model, because it is capable of creating a loss function that estimate how “real” the image look, instead of more simple pixel to pixel loss functions. 



## Architecture

  <img src="https://github.com/SalomonMalka/Black-and-white-image-colorization/blob/main/Resources/unet.png" height="300">
  <p align="center">

## Loss functions
The most standard losses are MSE and L1 losses. 
However, there are not a good choice when comparing images: there do not consider high level features

<img src="https://github.com/SalomonMalka/Black-and-white-image-colorization/blob/main/Resources/exemple_mse.png" height="300">
  <p align="center">


There are a bunch of losses that compare images in a way that is closer to how human perceive images. 
The losses we chose are:
LPIPS (LEARNED PERCEPTUAL IMAGE PATCH SIMILARITY)
Perceptual loss
VGG
They are essentially pretrained freezed networks that extract high level features of each images. The loss is the MSE loss between the features values



## Further development ideas

1. Training with bigger dataset and on more epochs
2. Try different perceptual loss functions to see if training times can be shorter.
3. Use a pretrained semantic segmentation network and apply it before the input. This should considerably improve the results as the colorization is done with the semantic component information of the image

## References

- The article we used as a baseline for our project: [Colorizing black & white images with U-Net and conditional GAN — A Tutorial
](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)
- A paper that was very usfull to us: [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)
- Another paper that was usfull to us: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- Another article on the colorization problem: [Image Colorization with Convolutional Neural Networks](https://lukemelas.github.io/image-colorization.html)


