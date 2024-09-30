# Signal Classification using Preserved CNN Model

## Project Overview
This project focuses on the classification of time-series signals using a Convolutional Neural Network (CNN). The CNN model is preserved for future use and integrated into a user-friendly GUI. The model is trained on pre-processed signal data and evaluated using various performance metrics such as accuracy, precision, recall, F1 score, specificity, and ROC AUC.

## Key Features
- **Model Architecture**: Utilizes a 1D CNN model designed for binary classification of time-series signals.
- **Training Process**: The model has trained over 35 epochs with callbacks like early stopping and a learning rate scheduler to prevent overfitting and improve convergence.
- **Evaluation**: The model is evaluated using metrics such as confusion matrix, ROC curve, accuracy, precision, recall, F1 score, and specificity.
- **Unseen Data Testing**: The preserved model is tested on random unseen data samples to validate its real-world applicability.
- **Training and Validation Curves**: Visualizations include loss and accuracy curves to monitor the model’s performance across epochs.

## GUI Application
The preserved model is integrated into a GUI built using **Tkinter**, providing a user-friendly interface where users can:
- **Upload Signal Data**: Users can upload signal data in batch for processing.
- **Plot and Visualize Signals**: The GUI allows users to plot signals from uploaded data.
- **Predict Signals**: Users can run predictions on uploaded signals to determine their classification.
- **Save Model**: The GUI also supports saving the model for future use.

## Author Contribution
Author Bidhan Paul and co-author Fahim Talukdar contributed equally to this work. Author Bidhan Paul led the development of the model and its integration into the GUI. Author Fahim Talukdar managed the data pre-processing and performed extensive testing of the preserved model. Both authors were involved in the analysis, writing, and final presentation of the results.

## Conclusion
The project demonstrates the effectiveness of CNN in signal classification tasks and provides a practical tool for users to classify signals via an easy-to-use GUI. The preserved model allows for continuous use and integration into other applications, while the evaluation metrics confirm the robustness of the model across various datasets.
