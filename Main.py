import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from CNN import CNNModel  # Import CNNModel from CNN.py
from RNN import RNNModel  # Import RNNModel from RNN.py

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Clean text data: Remove special characters and lowercase text
    data['clean_text'] = data['clean_text'].str.replace('[^a-zA-Z\s]', '').str.lower()
    
    # Check for labels outside the range [-1, 0, 1]
    valid_labels = [-1, 0, 1]  # Adjust this list based on your labels
    
    # Remove rows where the 'category' column has a value outside the valid range
    data = data[data['category'].isin(valid_labels)]
    
    return data


# ... (other code remains the same)

def prepare_data(data):
    # Fill NaN values in 'text' column with empty strings
    data['clean_text'].fillna('', inplace=True)

    # Encode sentiment labels to numerical values
    label_encoder = LabelEncoder()
    data['category'] = label_encoder.fit_transform(data['category'])

    # Convert text data into numerical representations using Tokenizer
    tokenizer = Tokenizer(num_words=10000)  # Limit vocabulary size to 10000 words

    tokenizer.fit_on_texts(data['clean_text'].astype(str))  # Ensure data is treated as strings
    sequences = tokenizer.texts_to_sequences(data['clean_text'].astype(str))  # Convert to strings for tokenization
    max_sequence_length = max([len(seq) for seq in sequences])
    X_padded = pad_sequences(sequences, maxlen=max_sequence_length)

    return X_padded, data['category'], max_sequence_length

# ... (rest of the code remains the same)


def split_dataset(X, y, test_size=0.2, validation_size=0.25):
    # Split dataset into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test






def main():
    # Load the dataset
    file_path = 'Twitter_Data 2.csv'  # Update with your dataset file path
    data = load_data(file_path)

    # Data Preprocessing
    data = preprocess_data(data)

    # Data Preparation
    X, y, max_sequence_length = prepare_data(data)

    # Split dataset into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Number of test samples: {len(X_test)}")
    print(f"Max sequence length: {max_sequence_length}")




     # Train and evaluate CNN model
    cnn_model = CNNModel()
    cnn_model.train(X_train, y_train, X_val, y_val)
    cnn_model.evaluate(X_test, y_test)

    # Train and evaluate RNN model
    rnn_model = RNNModel()
    rnn_model.train(X_train, y_train, X_val, y_val)
    rnn_model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
