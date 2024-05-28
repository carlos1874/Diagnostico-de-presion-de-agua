import pandas as pd 
import numpy as np 

class DataProcessor: 
    def __init__(self, file_path, columns_to_convert): 
        self.file_path = file_path 
        self.columns_to_convert = columns_to_convert 
        self.data = None 

    def load_data(self): 
        self.data = pd.read_excel(self.file_path) 

    def clean_data(self): 
    # Eliminar espacios extra en las cadenas de texto 
        self.data = self.data.replace(r'\s+', ' ', regex=True).replace(' ', np.nan) 
        self.data.replace(to_replace=[r'^\s*#\s*N/A\s*$', '#N/A', np.nan], value=np.nan, regex=True, inplace=True) 
        self.data.dropna(inplace=True) 

    def convert_columns_to_float(self): 
        for col in self.columns_to_convert: 
            try: 
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce') 
            except Exception as e: 
                print(f"Error al convertir la columna {col}: {e}") 
            continue  # Puede usar 'continue' para seguir con la siguiente columna 
        self.data[self.columns_to_convert] = self.data[self.columns_to_convert].astype(float) 

    def normalize_data(self): 
        self.data[self.columns_to_convert] = (self.data[self.columns_to_convert] - self.data[self.columns_to_convert].min()) / (self.data[self.columns_to_convert].max() - self.data[self.columns_to_convert].min()) 

    def get_features_and_labels(self): 
        X = self.data[['Elevation(m)', 'Base Demand(LPS)', 'Demand(LPS)', 'Head(m)', 'Pipe Diameter(mm)', 'Pipe Length(m)', 'Pipe Roughness(mm)', 'Flow in Pipe(LPS)', 'Water Velocity(m/s)', 'Valve Status']].values 
        y = self.data['Pressure(m)'].values 
        return X, y 

 

    def save_processed_data(self, output_path): 

        self.data.to_excel(output_path, index=False) 

