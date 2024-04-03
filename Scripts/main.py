# Import libraries for data handling and manipulation
import numpy as np
import random
import json
import sys

# Import Keras libraries for building the model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, RepeatVector, TimeDistributed
from keras.utils import to_categorical
import keras

# Define a function to generate a dataset with binary multiplication examples
def create_dataset(size, bits):
    max_val = 2**bits
    dataset_A, dataset_B, dataset_C = [], [], []
    
    for _ in range(size):
        # Generate random integers and convert them to binary format
        num_A = random.randint(0, max_val - 1)
        num_B = random.randint(0, max_val - 1)
        num_C = num_A * num_B

        bin_A = format(num_A, f'0{bits}b')[::-1]
        bin_B = format(num_B, f'0{bits}b')[::-1]
        bin_C = format(num_C, f'0{bits*2}b')[::-1]

        # Append the binary numbers to their respective lists
        dataset_A.append(bin_A)
        dataset_B.append(bin_B)
        dataset_C.append(bin_C)
        
    return dataset_A, dataset_B, dataset_C

# Define a class for one-hot encoding and decoding of binary strings
class BinaryOneHotEncoder:
    def __init__(self):
        self.char_to_index = {'0': 0, '1': 1}
        self.index_to_char = {0: '0', 1: '1'}

    def encode(self, binary_str):
        return to_categorical([self.char_to_index[char] for char in binary_str], num_classes=2)

    def decode(self, encoded_array, include_junk=False, total_length=16):
        if include_junk:
            decoded = ''.join(self.index_to_char[np.argmax(vec)] for vec in encoded_array)[1:]
        else:
            decoded = ''.join(self.index_to_char[np.argmax(vec)] for vec in encoded_array)
        return decoded.zfill(total_length)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python main.py --param param/param.json --train-size 10000 --test-size 1000 --seed 1234")
        sys.exit()
    
    # Parse command line arguments
    params = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv), 2)}
    
    # Load hyperparameters from JSON file
    with open(params.get("--param", "param/param.json")) as file:
        hyperparameters = json.load(file)
    
    # Set seed for reproducibility
    np.random.seed(int(params.get("--seed", 1234)))

    # Generate datasets
    print("Generating datasets...")
    train_A, train_B, train_C = create_dataset(int(params.get("--train-size", 10000)), hyperparameters['bin_dim'])
    test_A, test_B, test_C = create_dataset(int(params.get("--test-size", 1000)), hyperparameters['bin_dim'])
    
    encoder = BinaryOneHotEncoder()
    
    # Prepare data for training
    x_train = np.array([encoder.encode(a + b) for a, b in zip(train_A, train_B)])
    y_train = np.array([encoder.encode(c) for c in train_C])
    x_test = np.array([encoder.encode(a + b) for a, b in zip(test_A, test_B)])
    y_test = np.array([encoder.encode(c) for c in test_C])

    # Build the model
    model = Sequential([
        LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2]), activation='tanh', recurrent_activation='sigmoid'),
        RepeatVector(y_train.shape[1]),
        LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'),
        TimeDistributed(Dense(2, activation='softmax'))
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=hyperparameters['learning rate']), metrics=['accuracy'])

    # Train the model
    print("Training model...")
    model.fit(x_train, y_train, batch_size=hyperparameters['batch_size'], epochs=hyperparameters['num iter'], validation_data=(x_test, y_test))

    # Test model performance on a subset of the test data
    print("Evaluating model...")
    predictions = model.predict(x_test[:100])
    for i, (pred, true) in enumerate(zip(predictions, y_test[:100])):
        print(f"Example {i+1}:")
        print("Predicted:", encoder.decode(pred, include_junk=True))
        print("True:", encoder.decode(true, include_junk=True))
        print()
