import sys
import numpy
import pandas
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_train_test_data(train_ratio=.7):
    data = pandas.read_csv('./stats.csv')
    feature_col = ['POS', 'AGE', 'MIN%', 'USG%', 'TO%', 'FT%', '2P%', '3P%', 'eFG%', 'TS%',
                   'PPG', 'RPG', 'TRB%', 'APG', 'AST%', 'SPG', 'BPG', 'TOPG', 'VI', 'ORTG', 'DRTG']
    label_col = ['SALARY']
    X = data[feature_col]
    y = data[label_col]

    return sklearn.model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=0)


def load_precise_train_test_data(train_ratio=.7):
    precise_data = pandas.read_csv('./stats.csv')
    feature_col = ['AGE', 'PPG', 'APG', 'RPG', 'SPG']
    label_col = ['SALARY']
    X = precise_data[feature_col]
    y = precise_data[label_col]

    return sklearn.model_selection.train_test_split(X, y, test_size=1 - train_ratio, random_state=0)


def train_predict(X_train, X_test, y_train, y_test, precise):
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

    # picture
    for i in range(len(X_train.columns)):
        plt.scatter(X_train.iloc[:, i], y_train['SALARY'], label='train')
        plt.scatter(X_test.iloc[:, i], y_test['SALARY'],
                    label='test', color='green')
        plt.scatter(X_test.iloc[:, i], y_pred, label='predict', color='red')
        plt.title("NBA Salary Prediction")
        plt.legend(loc='upper left')
        plt.xlabel(str(X_train.columns[i]))
        plt.ylabel('Ten Million Dollors')
        if precise:
            plt.savefig(str(X_train.columns[i])+"_precise.png")
        else:
            plt.savefig(str(X_train.columns[i])+".png")
        plt.clf()
    return lr


def salary_prediction(lr, age, ppg, apg, rpg, spg):
    player = [
        [age, ppg, apg, rpg, spg]
    ]
    df = pandas.DataFrame(player, columns=['AGE', 'PPG', 'APG', 'RPG', 'SPG'])
    salary = lr.predict(df)
    print("你預測在NBA會拿到的薪水為：$", int(salary[0, 0]))


def main(argv):
    # using all data
    X_train, X_test, y_train, y_test = load_train_test_data(train_ratio=.7)
    X_train_precise, X_test_precise, y_train_precise, y_test_precise = load_precise_train_test_data(
        train_ratio=.7)

    # using precise data
    lr = train_predict(X_train, X_test, y_train, y_test, precise=False)
    lr = train_predict(X_train_precise, X_test_precise,
                       y_train_precise, y_test_precise, precise=True)

    # input
    age = input('AGE: ')
    ppg = input('PPG: ')
    apg = input('APG: ')
    rpg = input('RPG: ')
    spg = input('SPG: ')
    salary_prediction(lr, float(age), float(
        ppg), float(apg), float(rpg), float(spg))


if __name__ == "__main__":
    main(sys.argv)
