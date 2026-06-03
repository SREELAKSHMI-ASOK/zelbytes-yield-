import pandas as pd
import numpy as np

def generate_sample_row():
    sample_data = {
        'Temperature (°C)': round(np.random.uniform(18.0, 24.0), 1),
        'Humidity (%)': round(np.random.uniform(85.0, 95.0), 1),
        'CO2 (ppm)': int(np.random.uniform(800, 1200)),
        'Yield (kg/m²)': round(np.random.uniform(1.5, 4.0), 2)
    }
    return pd.DataFrame([sample_data])

if __name__ == '__main__':
    print('--- Environment Check: SUCCESS ---')
    print('Sample Polyhouse Sensor Data:')
    print(generate_sample_row().to_string(index=False))