import sys
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_train_test_data(train_ratio):
    data = pandas.read_csv('./train_test.csv')
    feature_col = ['AGE', 'MIN%', 'USG%', 'TO%', 'FT%', '2P%', '3P%', 'eFG%', 'TS%',
                   'PPG', 'RPG', 'TRB%', 'APG', 'AST%', 'SPG', 'BPG', 'TOPG', 'VI', 'ORTG', 'DRTG']
    label_col = ['SALARY']
    X = data[feature_col]
    y = data[label_col]

    return sklearn.model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=0)


def load_players_data():
    data = pandas.read_csv('./players.csv')
    feature_col = ['AGE', 'MIN%', 'USG%', 'TO%', 'FT%', '2P%', '3P%', 'eFG%', 'TS%',
                   'PPG', 'RPG', 'TRB%', 'APG', 'AST%', 'SPG', 'BPG', 'TOPG', 'VI', 'ORTG', 'DRTG']
    name_col = ['NAME']
    label_col = ['SALARY']
    X = data[feature_col]
    y = data[label_col]
    name = data[name_col]

    return X, y, name


def train_predict(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    # score
    print("train score: ", lr.score(X_train, y_train))
    print("test score: ", lr.score(X_test, y_test))
    print("------")

    # loss function
    print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('RMSE:', numpy.sqrt(
        metrics.mean_squared_error(y_test, y_pred)))
    print("------")

    # draw picture
    for i in range(len(X_train.columns)):
        plt.scatter(X_train.iloc[:, i], y_train['SALARY'], label='train')
        plt.scatter(X_test.iloc[:, i], y_test['SALARY'],
                    label='test', color='green')
        plt.scatter(X_test.iloc[:, i], y_pred, label='predict', color='red')
        plt.title("NBA Salary Prediction")
        plt.legend(loc='upper left')
        plt.xlabel(str(X_train.columns[i]))
        plt.ylabel('Ten Million Dollors')
        plt.savefig("./stats_pic/"+str(X_train.columns[i])+".png")
        plt.clf()
    return lr


def predict_salary(lr, X, y, name):
    salary = lr.predict(X)
    for i in range(len(name)):
        print(name.iloc[i, 0], "原本薪水為：$", y.iloc[i, 0],
              "\t預測薪水為：$", int(salary[i, 0]))


def main(argv):
    # load and split data
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.7)
    lr = train_predict(X_train, X_test, y_train, y_test)

    # using data to predict salary
    X, y, name = load_players_data()
    predict_salary(lr, X, y, name)


if __name__ == "__main__":
    main(sys.argv)
