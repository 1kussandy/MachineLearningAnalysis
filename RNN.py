import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

class RNNModel:
    def __init__(self, max_sequence_length=49, max_words=10000, embedding_dim=100):
        self.max_sequence_length = max_sequence_length
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.max_sequence_length))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(3, activation='softmax'))  # 3 classes: negative, neutral, positive

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, y_val), callbacks=[early_stopping])
        
        # Evaluate model on validation set and print accuracy
        _, accuracy = self.model.evaluate(X_val, y_val)
        print(f"RNN Validation Accuracy: {accuracy}")
        print(f"RNN Validation Accuracy: {accuracy * 100:.2f}%")

        
        return history

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"RNN Test Accuracy: {accuracy}")
        print(f"RNN Test Accuracy: {accuracy * 100:.2f}%")

        return accuracy