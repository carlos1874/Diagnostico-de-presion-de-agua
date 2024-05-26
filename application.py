import numpy as np
import tensorflow as tf
from data_processor import DataProcessor
from neural_network_model import NeuralNetworkModel

def main():
    # Procesamiento de datos
    file_path = 'C:/Users/Usuario/Desktop/Preassure_water/data/dataset.xlsx'
    output_path = 'dataset_process.xlsx'
    columns_to_convert = ['Elevation(m)', 'Base Demand(LPS)', 'Demand(LPS)', 'Head(m)', 'Pressure(m)']
    
    processor = DataProcessor(file_path, columns_to_convert)
    processor.load_data()
    processor.clean_data()
    processor.convert_columns_to_float()
    processor.normalize_data()
    processor.save_processed_data(output_path)

    X, y = processor.get_features_and_labels()

    # Entrenamiento de la red neuronal
    model = NeuralNetworkModel()
    model.build_model(input_dim=X.shape[1])
    model.train_model(X, y)
    tf.keras.models.save_model(model.model, 'my_model.keras')


    # Predicción de presión
    elevation = float(input("Ingrese la elevación (m): "))
    base_demand = float(input("Ingrese la demanda base (LPS): "))
    demand = float(input("Ingrese la demanda (LPS): "))
    head = float(input("Ingrese la cabeza (m): "))

    loaded_model = tf.keras.models.load_model('my_model.keras')
    input_data = np.array([[elevation, base_demand, demand, head]])
    pressure = model.predict(input_data)
    print(f"La presión estimada es: {pressure:.2f} m")

if __name__ == "__main__":
    main()
