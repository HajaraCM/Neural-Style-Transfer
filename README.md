# Neural Style Transfer using VGG19 and TensorFlow Hub

## Overview

This project demonstrates the application of **Neural Style Transfer (NST)**, a technique that allows us to transform an image by blending the **content** of one image with the **style** of another. The project explores two different approaches:

1. **Using VGG19**: This approach involves using a pre-trained Convolutional Neural Network (CNN) model to extract content and style features, followed by optimization techniques to combine them.
2. **Using TensorFlow Hub**: A simplified, pre-trained model from TensorFlow Hub to apply style transfer directly with minimal setup.

## Key Concepts

- **Content Image**: The image whose content (objects, structure, and layout) will be retained in the output.
- **Style Image**: The image whose artistic style (textures, colors, and patterns) will be applied to the content image.
- **Neural Style Transfer (NST)**: The process of extracting features from both the content and style images and blending them together using a loss function, typically optimized using gradient descent.

## Tools & Libraries

- **VGG19**: A pre-trained Convolutional Neural Network (CNN) used to extract content and style features.
- **TensorFlow**: A deep learning framework used for implementing and running the models.
- **TensorFlow Hub**: A library with pre-trained models, including a style transfer model.
- **NumPy**: For handling image data and mathematical operations.
- **Matplotlib**: For visualizing the images during and after the process.

![Screenshot 2024-12-03 152211](https://github.com/user-attachments/assets/032296e4-be47-40e7-b52f-a6d7b89cd277)


## Approach

### 1. Neural Style Transfer with VGG19

- **VGG19 Model**: This model is a deep neural network trained on large image datasets like ImageNet. It is used here to extract **content features** and **style features** from the content and style images, respectively.
  
- **Feature Extraction**: 
  - **Content Features**: Extracted from the deeper layers of the VGG19 model, which capture high-level representations like object shapes and structures.
  - **Style Features**: Extracted from earlier layers that capture low-level patterns, textures, and colors in the style image.
  
- **Loss Function**: The generated image is optimized to minimize two loss functions:
  - **Content Loss**: Measures how much the content of the generated image deviates from the content image.
  - **Style Loss**: Measures how much the style of the generated image deviates from the style image.

- **Optimization**: A gradient descent algorithm is used to update the generated image iteratively, minimizing the total loss (content + style) to achieve a blend of the two images.

### 2. Neural Style Transfer with TensorFlow Hub

- **Pre-trained Model**: TensorFlow Hub provides a pre-trained style transfer model, making it easy to apply style transfer without needing to worry about the underlying complexity.
  
- **Model Functionality**: The TensorFlow Hub model takes two images (content and style) as inputs and outputs a new image that combines the content of the first image and the style of the second image.
  
- **Ease of Use**: By using the pre-trained model, we can quickly apply different styles to various content images with minimal configuration, making this approach more accessible for rapid experimentation.



## Workflow

1. **Preprocessing**:
   - The content and style images are resized to a standard size that the model expects (typically 256x256 or 224x224 pixels).
   - The images are normalized and converted to the appropriate format (usually floating-point).

2. **Feature Extraction** (For VGG19 Approach):
   - The VGG19 model extracts **content** and **style** features from the images.
   - These features are used to compute the loss functions, guiding the optimization of the generated image.

3. **Optimization**:
   - Using a gradient descent algorithm, the generated image is iteratively updated to reduce the loss.
   - The loss is a combination of content loss and style loss, which ensures that the generated image retains both the content of the content image and the style of the style image.

4. **Result**:
   - The result is a new image that contains the content of the original image and the style of the second image.



## Conclusion

This project demonstrates the power of deep learning to merge art and technology. Using the **VGG19 model**, we explored how neural networks can be used to separate and recombine the content and style of images. The **TensorFlow Hub** model simplifies this process, offering a straightforward way to apply style transfer using pre-trained models.

Both methods showcase the versatility and creativity enabled by machine learning in the domain of image processing. The VGG19 approach offers more control over the process, while TensorFlow Hub provides a quick and efficient method to achieve style transfer.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

