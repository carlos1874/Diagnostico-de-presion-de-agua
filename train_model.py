import matplotlib.pyplot as plt
from data_processor import DataProcessor
from neural_network_model import NeuralNetworkModel

def train_and_save_model():
    # Procesamiento de datos
    file_path = 'C:/Users/Usuario/Desktop/Preassure_water/data/dataset.xlsx'
    columns_to_convert = ['Elevation(m)', 'Base Demand(LPS)', 'Demand(LPS)', 'Head(m)', 'Pressure(m)']
    
    processor = DataProcessor(file_path, columns_to_convert)
    processor.load_data()
    processor.clean_data()
    processor.convert_columns_to_float()
    processor.normalize_data()

    X, y = processor.get_features_and_labels()

    # Entrenamiento de la red neuronal
    model = NeuralNetworkModel()
    model.build_model(input_dim=X.shape[1])
    history = model.train_model(X, y)  # Recibir el historial aquí
    model.save_model('my_model_v2.keras')

    # Graficar las pérdidas
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('path_to_save_figure.png')      

if __name__ == "__main__":
    train_and_save_model()
