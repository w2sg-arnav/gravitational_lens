# Gravitational Lens Classifier with PyTorch

This repository contains Jupyter Notebooks that implement a deep learning model (specifically, a ResNet18) to classify astronomical images as either containing a gravitational lens or not.  The project addresses a significant class imbalance in the dataset.

## Task Description

**Specific Test II. Lens Finding**

**Task:** Build a model to identify gravitational lenses using PyTorch.

**Dataset:** The dataset is comprised of observational data of strong lenses and non-lensed galaxies.  Images are provided in three different filters for each object, resulting in an array shape of (3, 64, 64) for each object.

*   **Training Data:** Located in the `train_lenses` and `train_nonlenses` directories.
*   **Evaluation Data:** Located in the `test_lenses` and `test_nonlenses` directories.
*   **Class Imbalance:** The number of non-lensed galaxies is significantly larger than the number of lensed galaxies. This imbalance is a key challenge addressed in the implementation.

**Evaluation Metrics:**

*   ROC curve (Receiver Operating Characteristic curve)
*   AUC score (Area Under the ROC Curve)

**Dataset Link:**

[https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view?usp=drive_link](https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view?usp=drive_link)

## File Structure
```
gravitational_lens/
├── data/
│ └── Specific_test_2/ <- Unzip the dataset here
│ ├── train_lenses/ <- Training images with lenses
│ ├── train_nonlenses/ <- Training images without lenses
│ ├── test_lenses/ <- Testing images with lenses
│ └── test_nonlenses/ <- Testing images without lenses
├── notebooks/
│ ├── classification_notebooks.ipynb <- Main notebook (using SimpleCNN)
│ └── classification_notebooks_resnet.ipynb <- Main notebook (using ResNet18 - RECOMMENDED)
├── models/ <- Directory to save the trained model (will be created)
│ └── lens_finder_model.pth <- Trained model weights (created after training)
├── README.md <- This file
└── requirements.txt <- Python dependencies
```
**Important:**  After downloading the dataset from the Google Drive link, unzip it directly into the `data/Specific_test_2/` directory.  The notebook expects the `train_lenses`, `train_nonlenses`, `test_lenses`, and `test_nonlenses` directories to be *directly* under `data/Specific_test_2/`, *not* nested within further `train` and `test` subdirectories. The final path should be `data/Specific_test_2/train_lenses`, etc., and *not* `data/Specific_test_2/train/train_lenses`.

## Notebooks

*   **`classification_notebooks_resnet.ipynb` (Recommended):** This is the primary notebook and uses a pre-trained ResNet18 model for classification.  This notebook is thoroughly debugged, addresses the class imbalance effectively, and is ready to run.  It includes:
    *   Data loading and preprocessing (including per-channel normalization).
    *   Data augmentation (random rotations, horizontal/vertical flips, affine transformations).
    *   Definition of the ResNet18 model (with the final fully connected layer modified for binary classification).
    *   Training loop with:
        *   Weighted random sampling to address class imbalance.
        *   BCEWithLogitsLoss (binary cross-entropy with logits) as the loss function.
        *   Adam optimizer.
        *   ReduceLROnPlateau learning rate scheduler.
        *   Progress bars using `tqdm`.
        *   Model saving (best model based on validation AUC).
    *   Evaluation on the test set (calculating and plotting ROC curve and AUC).
    *   Confusion matrix plotting.
    *   Training history plotting (loss and AUC vs. epochs).
    *   Extensive error handling and checks for empty/missing directories.
    *   Clear separation of train, validation and test sets

*   **`classification_notebooks.ipynb`:** This notebook uses a simpler CNN architecture. It can be used for comparison or as a starting point, but the `classification_notebooks_resnet.ipynb` notebook is strongly recommended for better performance.

## Setup and Running

1.  **Clone the Repository (Optional):** If you've cloned the repository, you can skip this step. If not, create the directory structure shown above.

2.  **Download the Dataset:** Download the dataset from the Google Drive link provided above.

3.  **Unzip the Dataset:** Unzip the downloaded file *directly* into the `data/Specific_test_2/` directory.  Ensure the four subdirectories (`train_lenses`, etc.) are *directly* under `data/Specific_test_2/`, as shown in the File Structure section.  *Do not create additional `train` or `test` subdirectories.*

4.  **Install Dependencies:**  Open a terminal or command prompt and navigate to the `gravitational_lens` directory (the top-level directory).  Then, install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

    This will install `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `pillow`, `numpy`, `seaborn`, and `tqdm`.

5.  **Open the Notebook:** Open either `classification_notebooks_resnet.ipynb` (recommended) or `classification_notebooks.ipynb` in Jupyter Notebook or JupyterLab.

6.  **Run the Notebook:** Run all cells in the notebook sequentially.  The notebook will:
    *   Load and preprocess the data.
    *   Create the `models` directory if it doesn't exist.
    *   Define and train the model (either ResNet18 or the simple CNN).
    *   Evaluate the model on the validation set during training.
    *   Save the best model weights (based on validation AUC) to `models/lens_finder_model.pth`.
    *   Evaluate the trained model on the test set.
    *   Print the test AUC.
    *   Display the ROC curve and confusion matrix.
    *   Plot the training history (loss and AUC).

7. **Run all the cells sequentially.**

**Key improvements and fixes:**

*   **Correct Data Loading:** The paths are now correct within the `LensDataset` and `create_dataloaders`.
*   **Class Imbalance:** The `WeightedRandomSampler` is now created *after* the train/validation split, using the correct class counts from the training subset.  A check for empty classes is included.
*   **Normalization:** Added per-channel normalization to the `LensDataset`.  This is *essential* for good performance, as the different filters likely have different intensity ranges.
*   **Error Handling:** Improved error handling in `LensDataset` to gracefully handle potential file loading errors and to provide a useful message
* **Data Augmentation:** Implements common data augmentation techniques (rotation, flipping, affine transforms) to improve model robustness.
*  **Model Output and Loss:** Made the output of the model have shape `(batch, 1)` and use `BCEWithLogitsLoss()`.  The labels are unsqueezed to match this. The sigmoid is moved outside the forward pass.
*   **Clear Comments and Docstrings:** Added comments and doc strings to explain the changes and improve overall code readability.
*   **Simplified dataset logic.**
*  **Added error and empty folder handling:** the code now handles more edge cases.
* **Directory and file verification:** I've added directory checks at the beginning of the notebook to ensure that the required data directories exist *before* attempting to load any data. This will catch common setup errors early on.
* **Model choice**: Made sure ResNet18 is uncommented and is used.
* **Correct function call**: Called `create_dataloaders` only once and in the correct place.
* **Corrected Indentation** Fixed Python indentation.

## Potential Further Improvements

*   **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, optimizer parameters (e.g., weight decay), and scheduler settings.
*   **More Complex Models:** If needed, try more complex ResNet architectures (ResNet50, ResNet101, etc.) or other pre-trained models (EfficientNet, DenseNet, etc.).
*   **Different Augmentations:** Explore other data augmentation techniques.
*   **Cross-Validation:** Use k-fold cross-validation for a more robust estimate of model performance.
*   **Error Analysis:** Carefully examine misclassified images to identify potential areas for improvement (data quality, model architecture, etc.).
* **Focal Loss** Using Focal loss might improve performance significantly.

This README provides a comprehensive guide to setting up and running the gravitational lens classification project.  The code is now robust, well-documented, and ready for further experimentation and refinement.

