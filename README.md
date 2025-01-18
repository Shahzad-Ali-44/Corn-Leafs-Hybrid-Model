# Corn Diseases Detection Using Hybrid Deep Learning Model (VGG-19 + ResNeXT)

## Overview
This project leverages a hybrid deep learning approach combining **VGG-19** and **ResNeXT** architectures to detect diseases in corn crops with a high accuracy of **94.7%**. By averaging predictions from both models, the hybrid model enhances reliability and performance, offering a robust solution for agricultural disease management.

## Features
- **Hybrid Approach**: Combines VGG-19 and ResNeXT for improved prediction accuracy.
- **High-Resolution Visualizations**: Confusion matrix and prediction grids for model evaluation.
- **Performance Metrics**: Detailed accuracy, loss plots, and a classification report.
- **Data Augmentation**: Robust preprocessing techniques to improve generalization.
- **Confidence Scores**: Each prediction includes a confidence percentage.

## Dataset
- **Source**: A curated dataset of corn crop images, including healthy and diseased samples.
- **Classes**: Multiple classes representing various diseases and healthy crops.
- **Preprocessing**:
  - Images are resized and normalized.
  - Augmentation techniques: rotation, shear, zoom, and horizontal flipping.

## Methodology
1. **Data Preparation**:
   - Augmentation applied to training and validation sets using `ImageDataGenerator`.
   - Batches of images processed for efficient model training.

2. **Model Architectures**:
   - **VGG-19**: Pre-trained on ImageNet, fine-tuned for corn disease detection.
   - **ResNeXT**: Pre-trained on ImageNet, optimized for advanced feature extraction.
   - Predictions from both models averaged for final hybrid results.

3. **Evaluation**:
   - Confusion matrix visualizations for class-wise accuracy.
   - Classification report providing precision, recall, and F1-score.

4. **Visualization**:
   - Accuracy and loss trends for both models and the hybrid approach.
   - Prediction grids showcasing actual vs. predicted labels with confidence scores.

## Dependencies
- Python
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Results
- **Hybrid Model Accuracy**: 94.7% on the test dataset.
- **Performance Metrics**:
  - High precision, recall, and F1-scores across all classes.
- **Visualizations**:
  - Confusion Matrix: Displays hybrid model predictions.
  - Accuracy & Loss Graphs: Tracks training progress.

## Visualizations
- **Confusion Matrix**: Highlights class-wise prediction accuracy.
- **Prediction Grid**: Shows actual vs. predicted labels for random test images.
- **Training Graphs**: Combined accuracy and loss trends for both models.

## Future Improvements
- Extend the dataset to include additional diseases and variations.
- Deploy the hybrid model on mobile or web platforms for real-time usage.
- Explore ensemble methods to incorporate more architectures for further accuracy improvements.

## Contributing
Contributions are welcome! Fork the repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the creators of the dataset.
- Gratitude to the TensorFlow/Keras community for their powerful tools.
- Acknowledgment to the open-source community for supporting this work.
