# Image Colorization Using GAN

Final Year Project 

[Complete Hardware Information](https://www.notion.so/Complete-Hardware-Information-65caf7e3087b4545bdfbe1931777a326)

- GPU Tesla P100
- GPU Memory 16GB
- RAM 13GB
- FastAI COCO_SAMPLE Dataset ( 1000 random images from Dataset )
- **Generator** - UNet
    - U-Net is a convolutional neural network architecture used for image segmentation tasks.
    - It consists of an encoder and a decoder.
    - The encoder downsamples the input image and extracts features.
    - The decoder upsamples the feature maps and generates a segmentation mask.
    - U-Net uses skip connections to connect the encoder and decoder layers.
    - The skip connections allow the decoder to use high-level features learned by the encoder to generate fine-grained segmentation masks.
    - U-Net is particularly effective for medical image segmentation tasks, such as segmenting tumors or organs from MRI or CT scans.
    - It has also been used for other image segmentation tasks, such as segmenting cells in microscopy images or segmenting buildings from satellite images.

    [Unet Architecture](https://www.notion.so/Unet-Architecture-35ada346e96940a3978a6e2ab37d5c57)

- **Discriminator**
    
    The discriminator architecture follows a simple approach. Blocks of Convolution-Batch Normalization-LeakyReLU layers are stacked together to discern whether the input image is real or fake. It should be noted that the first and last blocks do not incorporate normalization, while the last block has no activation function, since it is included in the loss function that will be used.
    
    In this implementation, we are using a "Patch" Discriminator. This type of discriminator outputs one number for every patch of a fixed size (in our case, 70 by 70 pixels) of the input image, and for each patch, it decides whether it is real or fake separately. This is different from a vanilla discriminator, which outputs one scalar value representing the overall authenticity of the input image.
    
    The patch discriminator is particularly useful for tasks such as colorization, where local changes are important and subtle details need to be taken into account. In our implementation, the discriminator's output shape is 30 by 30, but this does not mean that the patch size is 30 by 30. The actual patch size is determined by computing the receptive field of each of the 900 output numbers, which in our case is 70 by 70 pixels.
    
    [Patch Discriminator](https://www.notion.so/Patch-Discriminator-b705f4a2759f4a01b4875fd6426be63e)
    
- **GAN Loss**
    
    PyTorch implementation of the loss function for a Generative Adversarial Network (GAN). The loss function is used to train the generator and discriminator models in a GAN.
    
    The `GANLoss` class has two input parameters: `gan_mode` and `real_label`, and `fake_label`. The `gan_mode` parameter determines the type of GAN being used, either 'vanilla' or 'lsgan'. The `real_label` and `fake_label` parameters are the labels to use for real and fake images respectively.
    
    The `__init__` method registers the real and fake labels as buffers to be stored in the model. It also initializes the loss function to be used based on the `gan_mode` parameter.
    
    The `get_labels` method returns either the real or fake label depending on the `target_is_real` parameter. If `target_is_real` is `True`, then the real label is returned. Otherwise, the fake label is returned.
    
    The `__call__` method is the main method that calculates the loss. It calls the `get_labels` method to get the appropriate labels for the predictions. It then applies the loss function to the predictions and labels to calculate the loss. Finally, it returns the loss value.
    
- **Container :**
    
    `Sequential`: is a class in the PyTorch library that is used to define a feed-forward neural network. It is a container that allows users to define a sequence of layers in a neural network. The layers are added to the container in the order that they are passed to the `Sequential` constructor, and the input data is passed through the layers sequentially in the same order.
    
    `Sequential` provides an easy and convenient way to define a neural network in PyTorch, especially for simple architectures.
    
- **Layers :**
    - `Conv2d`: This is a convolutional layer that applies a set of filters to the input image to extract features. It performs a dot product between the weights of the filter and the corresponding pixels of the input image.
    - `BatchNorm2d`: This layer normalizes the input by subtracting the batch mean and dividing by the batch standard deviation. It helps with the internal covariance shift problem and makes the optimization process more stable.
    - `Dropout`: This layer randomly sets a fraction of the input to zero during training. It is used as a regularization technique to prevent overfitting.
    - `ConvTranspose2d`: This layer is used for upsampling or generating new images. It performs a reverse operation to convolution by mapping input pixels to multiple output pixels.

- **Activation functions :**
    - `ReLU`: This is a commonly used activation function that applies a threshold to the input values. It sets all negative values to zero and leaves positive values unchanged. This non-linear function helps to introduce non-linearity in the network and is widely used in deep neural networks.
- `LeakyReLU`: This is a variant of the ReLU activation function that introduces a small negative slope for negative input values. It helps to mitigate the "dying ReLU" problem and allows the network to learn from negative input values.
    
    The function is defined as `f(x) = max(x, alpha*x)`, where `alpha`is a hyperparameter that controls the slope of the function for negative input values.
    

- **Accuracy  :**
    
    > Accuracy is just a metrix to evaluate performance
    > 
    
    In a Generative Adversarial Network (GAN), accuracy is not a common evaluation metric since the goal is to generate realistic samples that can fool the discriminator, not to classify samples into pre-defined categories. 
    
    Instead, you can use other metrics such as the L1 or L2 loss, peak signal-to-noise ratio (PSNR), or structural similarity index (SSIM) to measure the similarity between the generated color image and the ground truth image.
    
    You can calculate the L1 or L2 loss between the generated color image and the ground truth image using the `nn.L1Loss` or `nn.MSELoss` loss functions provided by PyTorch. PSNR and SSIM can be calculated using third-party libraries such as `skimage` or `opencv`.
    
    It is important to note that evaluating the performance of a GAN model is not always straightforward, and metrics such as PSNR and SSIM may not always reflect the quality of the generated images. Therefore, it is recommended to also visually inspect the generated images to evaluate the performance of the model.