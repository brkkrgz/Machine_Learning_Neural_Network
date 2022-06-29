#####################################################################################################################
#   Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile, sep=',', names=['WKF', 'WKR', 'WRF', 'WRR', 'BKF', 'BKR', 'Depth'])

    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        self.processed_data = self.raw_input

        # setup dataframe for pandas
        self.df = pd.DataFrame(self.processed_data, columns=('WKF', 'WKR', 'WRF', 'WRR', 'BKF', 'BKR', 'Depth'))

        # preprocess categorical variables into numerical variables
        number = LabelEncoder()
        self.df['WKF'] = number.fit_transform(self.processed_data['WKF'].astype('str'))
        self.df['WRF'] = number.fit_transform(self.processed_data['WRF'].astype('str'))
        self.df['BKF'] = number.fit_transform(self.processed_data['BKF'].astype('str'))
        self.df['Depth'] = number.fit_transform(self.processed_data['Depth'].astype('str'))

        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.df.columns)
        nrows = len(self.df.index)
        X = self.df.iloc[:, 0:(ncols - 1)]
        y = self.df.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Below are the hyperparameters that you need to use for model
        #   evaluation
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics
        mlp = MLPRegressor(hidden_layer_sizes=(4, 4, 4), activation='relu', solver='adam', max_iter=200,
                           learning_rate_init=0.1)
        mlp2 = MLPRegressor(hidden_layer_sizes=(4, 4, 4), activation='tanh', solver='adam', max_iter=200,
                           learning_rate_init=0.1)
        mlp3 = MLPRegressor(hidden_layer_sizes=(4, 4, 4), activation='logistic', solver='adam', max_iter=200,
                           learning_rate_init=0.1)
        mlp.fit(X_train, y_train)

        predict_train = mlp.predict(X_train)
        predict_test = mlp.predict(X_test)

        rmse = math.sqrt(mean_squared_error(y_test, predict_test))
        r2 = r2_score(y_test, predict_test)

        print("The model performance for training set ReLu")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")

        mlp2.fit(X_train, y_train)
        predict_train = mlp2.predict(X_train)
        predict_test = mlp2.predict(X_test)

        rmse = math.sqrt(mean_squared_error(y_test, predict_test))
        r2 = r2_score(y_test, predict_test)

        print("The model performance for training set tanh")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")

        mlp3.fit(X_train, y_train)
        predict_train = mlp3.predict(X_train)
        predict_test = mlp3.predict(X_test)

        rmse = math.sqrt(mean_squared_error(y_test, predict_test))
        r2 = r2_score(y_test, predict_test)

        print("The model performance for training set logistic")
        print("--------------------------------------")
        print('RMSE is {}'.format(rmse))
        print('R2 score is {}'.format(r2))
        print("\n")

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        plt.plot(mlp.loss_curve_)
        plt.plot(mlp2.loss_curve_)
        plt.plot(mlp3.loss_curve_)
        plt.legend(['ReLu', 'Tanh', 'Logistic'], loc='upper left')
        plt.show()

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("https://ogunonu.github.io/krkopt.data") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
