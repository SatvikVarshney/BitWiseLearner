# RNN Binary Multiplication

## Overview
This project delves into the application of Recurrent Neural Networks (RNNs) to execute binary multiplication. Utilizing a custom dataset, the RNN model learns to multiply two binary numbers, offering insights into how deep learning can be applied to fundamental arithmetic operations. The focus is on understanding the binary multiplication process through a neural network's lens and evaluating the model's performance on both generated training and test datasets.

## Features
- **Binary Dataset Generation**: Dynamically creates datasets of binary numbers for training and testing the RNN model.
- **One-Hot Encoding**: Implements one-hot encoding to represent binary input and output data for the neural network.
- **RNN Model Training**: Employs a sequential model with LSTM layers tailored for sequential data processing, particularly suited for binary multiplication.
- **Performance Evaluation**: Assesses the model's accuracy by comparing predicted multiplication results against actual values in the test set.

## Getting Started

### Prerequisites
- Python 3.x
- Keras
- NumPy

### Data Files
The project automatically generates binary datasets according to specified parameters, including the size of training and test sets. Binary numbers are represented in "little endian" format, and the dataset comprises pairs of binary integers and their multiplication products.

### Configuration Files
Model hyperparameters and dataset configurations are stored in `.json` files within the `param` directory. These configurations include the learning rate, number of iterations for training, batch size, and the dimensions of binary numbers.

Example `param.json`:
```json
{
  "learning rate": 0.001,
  "num iter": 100,
  "batch_size": 64,
  "bin_dim": 8
}
```
### Installation
Clone this repository to get started with the project:

```bash
git clone https://github.com/yourusername/RNNBinaryMultiplication.git
```

Navigate to the project directory:
```bash
cd RNNBinaryMultiplication
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Usage
Run the project with the following command, specifying the parameters for the dataset size and random seed as needed:
```bash
python main.py --param param/param.json --train-size 10000 --test-size 1000 --seed 1234
```
