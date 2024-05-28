import numpy as np
import tensorflow as tf

# Cargar el modelo preentrenado
loaded_model = tf.keras.models.load_model('models/diagnostic_prediction.keras')

# Función para obtener y validar los datos de entrada del usuario
def get_input_data():
    data_labels = ['Elevation(m)', 'Base Demand(LPS)', 'Demand(LPS)', 'Head(m)', 'Pipe Diameter(mm)', 'Pipe Length(m)', 'Pipe Roughness(mm)', 'Flow in Pipe(LPS)', 'Water Velocity(m/s)', 'Valve Status']
    data_values = []

    for label in data_labels[:-1]:  # Excluir el último label que corresponde al estado de la válvula
        while True:
            try:
                value = float(input(f"Ingrese la {label}: "))
                data_values.append(value)
                break
            except ValueError:
                print(f"Entrada no válida para {label}. Por favor, ingrese un número válido.")

    # Solicitar el estado de la válvula
    valve_status = None
    while valve_status not in [1, 2, 3]:
        try:
            valve_status = int(input("Ingrese el estado de la válvula (1: Open, 2: Closed, 3: Partially Open): "))
            if valve_status not in [1, 2, 3]:
                print("Valor no válido. Por favor, ingrese 1 para 'Open', 2 para 'Closed' o 3 para 'Partially Open'.")
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número válido (1, 2 o 3) según el estado de la válvula.")

    data_values.append(valve_status)

    return np.array([data_values])

# Función para hacer la predicción usando el modelo cargado
def make_prediction(input_data):
    pressure = loaded_model.predict(input_data)
    return pressure[0][0] * 1.422  # Convertir a psi

# Función principal
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
