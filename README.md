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

- [Project goal and Motivation](#Project-goal-and-Motivation)
- [Repository Description](#repository-description)
- [Previous work](#Previous-work)
- [Architecture](#Architecture)
- [Loss functions](#Loss-functions)
- [Further development ideas](#further-development-ideas)
- [References](#References)
- [Notes](#Notes)

## Project goal and Motivation

The problem of colorizing an image is very interesting and was not possible to do in automatically before machine learning was introduced.
One of the biggest challenges in this problem is finding a loss function that

Our goal was Using different types of loss functions to find out which one works better for this task.


## Repository Description

| Filename                    | description                                                                                       |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `emotion-recognision.ipynb` | The main file in google colab format, including the prepossessing. to open import to google colab |
| `emotion-recognision.py`    | The main file in Python format                                                                    |
| `prepossessing.py`          | The prepossessing only in a python format                                                         |
| `dataset.py`                | Python file consists of the implementation of the dataset object.                                 |
| `architecture.py `          | Python file consists of the implementation of the proposed architecture.                          |
| `res `                      | Folder consists of all the images from the project                                                |
| `requirement.txt`           | File containing all the packages we used in this project                                          |
| `FinalProjectDeepLearning.pdf`           | The report of the project                                          |


## Previous work
We used an article about building Colorization algorithm as our base line for the project. 
Most of the sources we found use GAN architecture for their model, because it is capable of creating a loss function that estimate how “real” the image look, instead of more simple pixel to pixel loss functions. 

## Architecture

  <img src="https://github.com/SalomonMalka/Black-and-white-image-colorization/blob/main/Resources/unet.png" height="300">
  <p align="center">

## Loss functions


## Further development ideas

1. Try to expand our work to more facial expression datasets.
2. Try different uses of the attention mechanism.

## References

- The article we used as a baseline for our project: [Colorizing black & white images with U-Net and conditional GAN — A Tutorial
](https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8)
- A paper that was very usfull to us: [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)
- Another paper that was usfull to us: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
## Notes

- The data, and the predictor are too big to be uploaded to GitHub. You can found them in the following links:
  [train](https://drive.google.com/file/d/1wwtsQ1cCfpP132pGP7HZ5Ot7nmUZvimt/view?usp=sharing)
  [validation](https://drive.google.com/file/d/1q5qOGdZ0zkmZgv5Avyc1OrWa-FYNQX3S/view?usp=sharing),
  [test](https://drive.google.com/file/d/1pXyXMXUk08lZlnmqj-7hmD7xnKyM4Q7x/view?usp=sharing),
  [raw-images](https://drive.google.com/drive/folders/1FjyYvSZAEPQaoROEpr5FLtK2yGmLkt6x?usp=sharing),
  [predictor](https://drive.google.com/drive/folders/1o1DtnFnSwdRO8o23eW-a9jo_6cbY7ZA6?usp=sharing)
  (for the raw images and the predictor you need to download the files and put them in a folder with the exact same name as in the original folder)


