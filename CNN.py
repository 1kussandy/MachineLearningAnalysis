import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout

class CNN:
    def __init__(self, filepath):
        self.filepath = filepath
        self.model = None
        self.max_seq_length = None
        self.tokenizer = None
        self.label_mapping = None

    def load_data(self):
        df = pd.read_csv(self.filepath)
        return df

    def preprocess_data(self, df):
        tweets = df['tweets'].astype(str)
        tweets = tweets.str.lower()
    
        # Improved cleaning
        tweets = tweets.str.replace(r'http\S+|www.\S+', '', regex=True)  # Remove URLs
        tweets = tweets.str.replace(r'@\w+', '', regex=True)             # Remove mentions
        tweets = tweets.str.replace(r'#', '')                            # Keep hashtags text
        tweets = tweets.str.replace(r'[^a-zA-Z#\s]', '', regex=True)     # Keep letters and hashtags
    
        # Tokenization and padding
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(tweets)
        sequences = self.tokenizer.texts_to_sequences(tweets)
        self.max_seq_length = max(len(x) for x in sequences)
        tweets_padded = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post')
    
        # Encoding labels
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(df['sentiment'])
        labels = to_categorical(integer_encoded)
        self.label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    
        return tweets_padded, labels

    def build_model(self, vocab_size, embedding_dim=100, filter_sizes=128, kernel_size=5, num_classes=3):
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=self.max_seq_length))
        model.add(Conv1D(filter_sizes, kernel_size, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))  # Dropout to prevent overfitting
        model.add(Conv1D(filter_sizes // 2, kernel_size, activation='relu'))  # Additional Conv layer
        model.add(GlobalMaxPooling1D())
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.5))  # Additional Dropout layer
        model.add(Dense(num_classes, activation='softmax'))
    
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model
        
    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.7, random_state=42)

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, class_weight=None, callbacks=None):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), class_weight=class_weight, callbacks=callbacks)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss:.2f}')
        print(f'Test Accuracy: {accuracy*100:.2f}%')
        
    def get_sentiment_labels(self, prediction):
        predicted_index = np.argmax(prediction)
        sentiment_label = self.label_mapping[predicted_index]
        return sentiment_label
    
    def preprocess_new_data(self, new_tweets):
        new_tweets = new_tweets.str.lower()
        new_tweets = new_tweets.str.replace(r'http\S+|www.\S+', '', regex=True)
        new_tweets = new_tweets.str.replace(r'@\w+', '', regex=True)
        new_tweets = new_tweets.str.replace(r'#\w+', '', regex=True)
        new_tweets = new_tweets.str.replace(r'[^a-zA-Z\s]', '', regex=True)
        sequences = self.tokenizer.texts_to_sequences(new_tweets)
        tweets_padded = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post')
        return tweets_padded

    def predict(self, new_data):
        predictions = self.model.predict(new_data)
        return predictions
