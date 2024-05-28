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

    return np.array([data_values]), data_values

# Función para hacer la predicción usando el modelo cargado
def make_prediction(input_data):
    pressure = loaded_model.predict(input_data)
    return pressure[0][0] * 1.422  # Convertir a psi

# Función para diagnosticar la presión considerando múltiples variables
def diagnose_pressure(pressure, data_values):
    elevation, base_demand, demand, head, pipe_diameter, pipe_length, pipe_roughness, flow_in_pipe, water_velocity, valve_status = data_values
    
    diagnosis = []
    
    if pressure < 50:
        diagnosis.append("Presión muy baja.")
        if valve_status != 2:
            diagnosis.append("Posible fuga en el sistema o válvula parcialmente abierta cuando debería estar cerrada.")
        if flow_in_pipe < demand:
            diagnosis.append("Flujo en la tubería menor que la demanda. Posible obstrucción o fuga.")
    elif 50 <= pressure <= 150:
        diagnosis.append("Presión dentro del rango normal.")
    elif 150 < pressure <= 200:
        diagnosis.append("Presión alta.")
        if valve_status == 2:
            diagnosis.append("Válvula cerrada.")
        if pipe_diameter < 150:
            diagnosis.append("Diámetro de la tubería pequeño. Posible obstrucción.")
    else:
        diagnosis.append("Presión muy alta.")
        if head > 350:
            diagnosis.append("Cabeza alta. Posible bomba de presión demasiado alta.")
        if pipe_roughness > 0.2:
            diagnosis.append("Rugosidad de la tubería alta. Posible obstrucción severa.")

    return " ".join(diagnosis)

# Función principal
def main():
    while True:
        print("\n¿Desea realizar un diagnóstico? (S/N)")
        decision = input().strip().upper()
        if decision == 'S':
            input_data, data_values = get_input_data()
            estimated_pressure = make_prediction(input_data)
            diagnosis = diagnose_pressure(estimated_pressure, data_values)
            print(f"La presión estimada es: {estimated_pressure:.2f} psi")
            print(f"Diagnóstico: {diagnosis}")
        elif decision == 'N':
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, escriba 'S' para sí o 'N' para no.")

if __name__ == "__main__":
    main()
