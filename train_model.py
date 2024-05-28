import matplotlib.pyplot as plt 
from data_processor import DataProcessor 
from neural_network_model import NeuralNetworkModel 

def train_and_save_model(): 
    # Procesamiento de datos 
    file_path = '../Preassure_water/data/dataset.xlsx' 
    columns_to_convert = ['Elevation(m)', 'Base Demand(LPS)', 'Demand(LPS)', 'Head(m)', 'Pressure(m)', 'Pipe Diameter(mm)', 'Pipe Length(m)', 'Pipe Roughness(mm)', 'Flow in Pipe(LPS)', 'Water Velocity(m/s)', 'Valve Status'] 

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
    model.save_model('models/diagnostic_prediction.keras') 

    # Graficar las pérdidas 
    plt.plot(history.history['loss'], label='Train Loss') 
    plt.plot(history.history['val_loss'], label='Validation Loss') 
    plt.title('Model Loss During Training') 
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.legend() 

    plt.savefig('models/Grafica.png')       

if __name__ == "__main__": 
    train_and_save_model() 