#!/usr/bin/env python3
"""
14-visualize.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Eliminar columna
df = df.drop(columns=['Weighted_Price'])

# Cambiar nombre de Marca de tiempo
df = df.rename(columns={'Timestamp': 'Date'})

# Convertir marcas de tiempo a fecha y hora
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Índice de fecha
df = df.set_index('Date')

# Llenar el resto de los valores
df['Close'] = df['Close'].fillna(method='ffill')
df[['High', 'Low', 'Open']] = df[['High', 'Low', 'Open']].fillna(df['Close'])
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[
    ['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

# Filtrar datos desde 2017
df_2017 = df['2017':]

# intervalos diarios
df_daily = df_2017.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Mostrar grafica
df_daily.plot()
plt.show()

# Devuelve el DataFrame transformado
print(df)
