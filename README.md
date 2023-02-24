# Anime-DC-GAN

This project implements the DC GAN architecture from scratch to generate high-quality anime images. 
The model was trained on the Anime dataset for 500 epochs using an RTX 3060 6gb GPU. Tensorboard was used to visualize the quality of images.


## Getting Started

To get started with this project, you can follow these steps:

1. Clone the repository to your local machine
2. Download the Anime dataset and place it in a folder called data
3. Install the necessary libraries by running pip install -r requirements.txt
4. Train the model by running python train.py
5. Generate images using the trained model by running python generate.py

## Training

The model was trained for 500 epochs using an RTX 3060 6gb GPU for 6 hours. The loss and accuracy were monitored using Tensorboard, which can be accessed by running tensorboard --logdir logs. The generated images were also visualized using Tensorboard.

## Results

After training the model for 500 epochs, the generated anime images showed remarkable quality and clarity. These images can be viewed in the generated_images folder.

Following are few Cherry Picked Generated Images after training the model for 500 epochs:

![Generated Image 1](generated_images/image_epoch500.png/generated_7.png)
![Generated Image 2](generated_images/image_epoch500.png/generated_32.png)
![Generated Image 3](generated_images/image_epoch500.png/generated_8.png)
![Generated Image 1](generated_images/image_epoch500.png/generated_26.png)
![Generated Image 1](generated_images/image_epoch500.png/generated_45.png)

## Acknowledgments

This project was inspired by the paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Alec Radford, Luke Metz, and Soumith Chintala.

