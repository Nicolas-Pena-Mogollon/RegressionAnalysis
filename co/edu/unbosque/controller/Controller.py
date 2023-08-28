import numpy as np
from sklearn.preprocessing import MinMaxScaler

from co.edu.unbosque.model.CarPriceModel import train_model

linear_regression, scaler = train_model()



def main():
    'curbweight', 'horsepower', 'citympg', 'highwaympg'
    # new_instance = ['carwidth', 'carheight', 'curbweight', 'compressionratio', 'horsepower','peakrpm', 'citympg', 'highwaympg']
    new_instances = [[64.1, 48.8, 2548, 9, 111, 5000, 21, 27], [64.8, 54.3, 2395, 8.8, 111, 5800, 23, 29]]
    new_instances_scaled = scaler.transform(new_instances)
    print("Regression Results for each instance:")

    print(np.array2string(linear_regression.predict(new_instances_scaled), formatter={'all': lambda x: "{:.10f}".format(x)}))
