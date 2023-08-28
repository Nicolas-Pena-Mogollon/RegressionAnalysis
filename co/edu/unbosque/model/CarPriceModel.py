import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

motor_dataset = pd.read_csv(
    'https://docs.google.com/spreadsheets/d/e/2PACX-1vTPayo6DFI4aALxJlBRJOrgbRvxmH_htigpa3AEScoGxhjuZ0SywfGriUHSpianKGZc9Y88GzWx95cq/pub?gid=1563272139&single=true&output=csv')
x = []
y = []


def train_model():
    global x, y
    drop_unnecesary_fields()
    split_xy(tramsform_categorrized_data())  # Transform categorized data and then split x and y
    remove_low_variance_attributes()
    print("#jbjlas")

    y = remove_low_correlation_attributes()
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    return LinearRegression().fit(x, y), scaler


def drop_unnecesary_fields():
    motor_dataset.drop(['car_ID', 'CarName'], axis=1, inplace=True)


def tramsform_categorrized_data():
    return pd.get_dummies(motor_dataset,
                          columns=['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel',
                                   'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'])


def split_xy(data):
    global x, y
    y = data['price']
    x = data.drop(['price'], axis=1)


def remove_low_variance_attributes():
    global x
    selector = VarianceThreshold(threshold=20)
    x = selector.fit_transform(x)


def remove_low_correlation_attributes():
    global x
    y_float = pd.DataFrame({"price": y})

    # Eliminar los puntos y luego convertir la columna de cadenas a números de punto flotante
    y_float["price"] = y_float["price"].str.replace(".", "").astype(float)
    print(y_float["price"][0])

    selector = SelectKBest(f_regression, k=8)
    x = selector.fit_transform(x, y_float['price'].values)

    return y_float

