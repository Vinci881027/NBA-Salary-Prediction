import sys
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def load_train_test_data(train_ratio):
    data = pandas.read_csv('./stats.csv')
    feature_col = ['POS', 'AGE', 'MIN%', 'USG%', 'TO%', 'FT%', '2P%', '3P%', 'eFG%', 'TS%',
                   'PPG', 'RPG', 'TRB%', 'APG', 'AST%', 'SPG', 'BPG', 'TOPG', 'VI', 'ORTG', 'DRTG']
    label_col = ['SALARY']
    X = data[feature_col]
    y = data[label_col]

    # split by position
    position = {
        1: 'G',
        2: 'F',
        3: 'C'
    }
    data['position'] = data['POS'].map(position)
    enc = OneHotEncoder(categories=[['G', 'F', 'C']], sparse=False)
    deltas_arr = enc.fit_transform(
        data['position'].values.reshape(-1, 1))
    data['delta_G'] = deltas_arr[:, 0]
    data['delta_F'] = deltas_arr[:, 1]
    data['delta_C'] = deltas_arr[:, 2]
    data['delta_G_ppg'] = data['delta_G'] * data['PPG']
    data['delta_F_ppg'] = data['delta_F'] * data['PPG']
    data['delta_C_ppg'] = data['delta_C'] * data['PPG']
    X = data[['delta_G', 'delta_G_ppg', 'delta_F',
              'delta_F_ppg', 'delta_C', 'delta_C_ppg']].values
    y = data['SALARY'].values.reshape(-1, 1)

    return sklearn.model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(
        feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scaled = minmax_scaler.transform(X_train)
    X_test_scaled = minmax_scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    # w = []
    # for i in lr.coef_:
    #     for j in i:
    #         temp = []
    #         temp.append(j)
    #         w.append(temp)
    # w = numpy.array(w)

    # column_indices = [0, 2, 4]
    # positions = ['Guard', 'Forward', 'Center']
    # for i, position in zip(column_indices, positions):
    #     X_pred = X_train[X_train[:, i] == 1]
    #     y_pred = X_pred.dot(w)
    #     plt.scatter(X_pred[:, i + 1], y_pred, label=position)
    # plt.legend(loc='upper left')
    # plt.xlabel('Points Per Game')
    # plt.show()
    return lr


def predict(X_test, y_test, lr):
    # y_pred = lr.predict(X_test)
    column_indices = [0, 2, 4]
    positions = ['Guard', 'Forward', 'Center']
    for i, position in zip(column_indices, positions):
        X_pred = X_test[X_test[:, i] == 1]
        y_pred = lr.predict(X_pred)
        y_test_split = y_test[X_test[:, i] == 1]
        plt.scatter(X_pred[:, i + 1], y_test_split, label=position)
        plt.scatter(X_pred[:, i + 1], y_pred, label=position, color='black')
    plt.legend(loc='upper left')
    plt.xlabel('Scaled Points')
    plt.show()


def main(argv):
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, 0, 1)
    lr = train(X_train_scaled, y_train)
    predict(X_test_scaled, y_test, lr)


if __name__ == "__main__":
    main(sys.argv)
