## Text 2 Image - GAN

This project implements a Conditional Generative Adversarial Network (cGAN) for generating images, specifically trained on the CUB_200_2011 birds dataset. The implementation is based on TensorFlow and Keras.

### Dataset

The dataset used for training is the CUB_200_2011 birds dataset. The dataset files are provided in the `dataset` folder. Due to the large size of the dataset files, they are not included in the repository. However, you can download the dataset files by running the `model.ipynb` notebook.

### Prerequisites

To set up the environment and install the required packages, follow these steps:

1. **Clone the Repository**

   ```sh
   git clone https://github.com/yourusername/conditional-gan-birds.git
   cd conditional-gan-birds
   ```

2. **Create a Virtual Environment**

   ```sh
   python -m venv myenv
   ```

3. **Activate the Virtual Environment**

   - On Windows:
     ```sh
     myenv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source myenv/bin/activate
     ```

4. **Install the Required Packages**

   Ensure you have `pip` installed. Then, install the packages from `requirements.txt`:

   ```sh
   pip install -r requirements.txt
   ```

### Running the Code

1. **Download the Dataset and Preprocess**

   Open the `model.ipynb` notebook and run all the cells to download the dataset and preprocess it for training.

2. **Train the Model**

   Running the cells in `model.ipynb` will also initiate the training process. The training involves the following steps:
   - Compiling and training the discriminator and generator models.
   - Logging the training process using TensorBoard.
   - Generating and saving sample images at regular intervals during training.

3. **Save the Model**

   The generator and discriminator models will be saved at the end of the training process. The saved models can be found in the root directory of the project.

### Structure of the Project

- `model.ipynb`: Jupyter notebook for downloading the dataset, preprocessing, training the model, and saving the trained models.
- `generator.h5`: The saved generator model after training.
- `discriminator.h5`: The saved discriminator model after training.
- `requirements.txt`: List of required packages for the project.
- `dataset`: Directory where the dataset files are stored.
- `test.ipynb`: To test the image generator model
- `toImportFunc.py`: Give necessary function to `test.ipynb` for testing of Generator

### Usage

- To train the model, run the `model.ipynb` notebook.
- To generate images using the trained generator, load the `generator.h5` model and use it to generate images from random noise and embedding vectors.

### Notes

- The training process might take a considerable amount of time depending on your hardware.
- Make sure your system meets the requirements for running TensorFlow and Keras.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

For any issues or contributions, feel free to open an issue or pull request on GitHub. Enjoy generating amazing bird images with your trained Conditional GAN!