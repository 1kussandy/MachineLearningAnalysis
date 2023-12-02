import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from CNN import CNNModel  # Import CNNModel from CNN.py
from RNN import RNNModel  # Import RNNModel from RNN.py

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data['clean_text'] = data['clean_text'].str.replace('[^a-zA-Z\s]', '').str.lower()
    valid_labels = [-1, 0, 1]  
    data = data[data['category'].isin(valid_labels)]
    return data


def prepare_data(data):
    data['clean_text'].fillna('', inplace=True)
    label_encoder = LabelEncoder()
    data['category'] = label_encoder.fit_transform(data['category'])
    tokenizer = Tokenizer(num_words=10000) 

    tokenizer.fit_on_texts(data['clean_text'].astype(str))  
    sequences = tokenizer.texts_to_sequences(data['clean_text'].astype(str))  
    max_sequence_length = max([len(seq) for seq in sequences])
    X_padded = pad_sequences(sequences, maxlen=max_sequence_length)

    return X_padded, data['category'], max_sequence_length

def split_dataset(X, y, test_size=0.2, validation_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test






def main():
    file_path = 'Twitter_Data 2.csv' 
    data = load_data(file_path)
    data = preprocess_data(data)
    X, y, max_sequence_length = prepare_data(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Number of test samples: {len(X_test)}")
    print(f"Max sequence length: {max_sequence_length}")

     # Train and evaluate CNN model
    cnn_model = CNNModel()
    cnn_model.train(X_train, y_train, X_val, y_val)
    cnn_accuracy=cnn_model.evaluate(X_test, y_test)

    # Train and evaluate RNN model
    rnn_model = RNNModel()
    rnn_model.train(X_train, y_train, X_val, y_val)
    rnn_accuracy = rnn_model.evaluate(X_test, y_test)

    # COMPARE HERE 
    print("\nComparison of CNN and RNN models:")
    print(f"CNN Validation Accuracy: {cnn_accuracy * 100:.2f}%")
    print(f"RNN Validation Accuracy: {rnn_accuracy * 100:.2f}%")
if __name__ == "__main__":
    main()
