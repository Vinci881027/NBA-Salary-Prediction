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


def load_train_test_data(train_ratio=.5):
    data = pandas.read_csv('./stats.csv')
    feature_col = ['POS', 'AGE', 'MIN%', 'USG%', 'TO%', 'FT%', '2P%', '3P%', 'eFG%', 'TS%',
                   'PPG', 'RPG', 'TRB%', 'APG', 'AST%', 'SPG', 'BPG', 'TOPG', 'VI', 'ORTG', 'DRTG']
    label_col = ['SALARY']
    X = data[feature_col]
    y = data[label_col]
    return sklearn.model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(
        feature_range=(low, upp)).fit(numpy.vstack((X_train, X_test)))
    X_train_scaled = minmax_scaler.transform(X_train)
    X_test_scaled = minmax_scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def main(argv):
    # X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.5)
    # X_train_scaled, X_test_scaled = scale_features(X_train, X_test, 0, 1)

    nba_data = pandas.read_csv('./stats.csv')
    position = {
        1: 'G',
        2: 'F',
        3: 'C'
    }
    nba_data['position'] = nba_data['POS'].map(position)
    enc = OneHotEncoder(categories=[['G', 'F', 'C']], sparse=False)
    deltas_arr = enc.fit_transform(
        nba_data['position'].values.reshape(-1, 1))
    nba_data['delta_G'] = deltas_arr[:, 0]
    nba_data['delta_F'] = deltas_arr[:, 1]
    nba_data['delta_C'] = deltas_arr[:, 2]
    nba_data['delta_G_ppg'] = nba_data['delta_G'] * nba_data['PPG']
    nba_data['delta_F_ppg'] = nba_data['delta_F'] * nba_data['PPG']
    nba_data['delta_C_ppg'] = nba_data['delta_C'] * nba_data['PPG']
    X = nba_data[['delta_G', 'delta_G_ppg', 'delta_F',
                  'delta_F_ppg', 'delta_C', 'delta_C_ppg']].values
    y = nba_data['SALARY'].values.reshape(-1, 1)

    # split train data and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # build and train model
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train, y_train)
    w = []
    for i in lr.coef_:
        for j in i:
            temp = []
            temp.append(j)
            w.append(temp)
    w = numpy.array(w)
    # draw picture
    column_indices = [0, 2, 4]
    positions = ['Guard', 'Forward', 'Center']
    for i, position in zip(column_indices, positions):
        X_pred = X_train[X_train[:, i] == 1]
        y_pred = X_pred.dot(w)
        plt.scatter(X_pred[:, i + 1], y_pred, label=position)
    plt.legend(loc='upper left')
    plt.xlabel('Points Per Game')
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
