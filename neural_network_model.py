import tensorflow as tf 
from tensorflow.keras.models import Sequential # type: ignore 
from tensorflow.keras.layers import Dense, Conv2D, Flatten # type: ignore 
from sklearn.model_selection import train_test_split 

class NeuralNetworkModel: 
    def __init__(self): 
        self.model = None 

    def build_model(self, input_dim): 
        self.model = Sequential() 
        self.model.add(Dense(64, input_dim=input_dim, activation='relu')) 
        self.model.add(Dense(32, activation='relu')) 
        self.model.add(Dense(1, activation='linear')) 
        self.model.compile(optimizer='adam', loss='mse') 

    def train_model(self, X, y, epochs=1000, batch_size=50, validation_split=0.2): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split) 
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split) 
        loss = self.model.evaluate(X_test, y_test) 
        print(f'Model Loss: {loss}') 
        return history 

    def save_model(self, file_path): 
        self.model.save(file_path) 

    def load_model(self, file_path): 
        self.model = tf.keras.models.load_model(file_path) 

    def predict(self, input_data): 
        prediction = self.model.predict(input_data) 
        return prediction[0][0] 

 