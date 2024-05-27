import numpy as np
import tensorflow as tf
from neural_network_model import NeuralNetworkModel

def get_input_data():
    elevation = float(input("Ingrese la elevación (m): "))
    base_demand = float(input("Ingrese la demanda base (LPS): "))
    demand = float(input("Ingrese la demanda (LPS): "))
    head = float(input("Ingrese la cabeza (m): "))
    return np.array([[elevation, base_demand, demand, head]])

def make_prediction(input_data):
    loaded_model = tf.keras.models.load_model('my_model_v2.keras')
    pressure = loaded_model.predict(input_data)[0][0]
    return pressure * 1.422  # Convert to psi

def main():
    while True:
        print("\n¿Desea realizar un diagnóstico? (S/N)")
        decision = input().strip().upper()
        if decision == 'S':
            input_data = get_input_data()
            estimated_pressure = make_prediction(input_data)
            print(f"La presión estimada es: {estimated_pressure:.2f} psi")
        elif decision == 'N':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, escriba 'S' para sí o 'N' para no.")

if __name__ == "__main__":
    main()
