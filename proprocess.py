import pandas as pd

data = pd.read_csv('train.csv', delimiter=',', usecols = ['Semana', 'Producto_ID', 'Cliente_ID', 'Demanda_uni_equil'],
                   dtype  = {'Semana': 'int32',
                             'Producto_ID':'int32',
                             'Venta_hoy':'float32',
                             'Venta_uni_hoy': 'int32',
                             'Dev_uni_proxima':'int32',
                             'Dev_proxima':'float32',
                             'Demanda_uni_equil':'int32'})

data = pd.pivot_table(data=data, values='Demanda_uni_equil', index=['Producto_ID', 'Cliente_ID'], columns='Semana', fill_value=0)

data.to_csv('train_pivot.csv')


