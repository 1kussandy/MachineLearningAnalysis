from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Class used for modular code

class CNNModel:
    # stores information related to the model architecture
    def __init__(self, max_sequence_length=49, max_words=10000, embedding_dim=100):
        self.max_sequence_length = max_sequence_length
        self.max_words = max_words
        self.embedding_dim = embedding_dim
        self.model = self.build_model()

    # builds the model with desired layers
    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.max_sequence_length))
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # trains the model on given datasets
    def train(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, y_val), callbacks=[early_stopping])
        
        _, accuracy = self.model.evaluate(X_val, y_val)
        print(f"CNN Validation Accuracy: {accuracy}")
        print(f"CNN Validation Accuracy: {accuracy * 100:.2f}%")

        
        return history

    # evaluates performance of the test data set
    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"CNN Test Accuracy: {accuracy}")
        print(f"CNN Test Accuracy: {accuracy * 100:.2f}%")

        return accuracy
