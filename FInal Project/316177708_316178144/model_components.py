"""
    Barak Bonker - 316177708
    Amit Avigdor -  316178144
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from entropymdlp import mdlp


def clean(file, classify):
    """
        clean the data and transfer the dtype to int or float if needed
        :param file: the data
        :param classify: the classify column name
    """
    file.dropna(subset=[classify])
    for column in file.columns:
        file[column] = file[column].replace('?', np.nan)
        if file[column].dtypes == "object":
            if file[column].str.contains('.').any():
                file[column] = pd.to_numeric(file[column], errors='ignore')



def fillByAll(file, classify):
    """
        fill missing values by the value of all rows
        :param file: the data
        :param classify: the classify column name
    """
    for column in file.columns:
        if column != classify:
            if file[column].isna().sum() > 0:
                if file[column].dtypes == "float64":
                    # if the column is float, fill the missing value by the mean of the column
                    file[column].fillna(file[column].mean(), inplace=True)
                else:
                    # if the column is int, fill the missing value by the most common value of the column
                    file[column].fillna(file[column].mode()[0], inplace=True)


def fillByClass(file, classify):
    """
        fill missing values by the values of the rows with the same classify value
        :param file: the data
        :param classify: the classify column name
    """
    for column in file.columns:
        if column != classify:
            if file[column].isna().sum() > 0:
                if file[column].dtypes == "float64":
                    # if the column is float, fill the missing value by the mean of the columns with the same classify value
                    file[column] = file.apply(
                        lambda row: file[file[classify] == row[classify]][column].mean() if pd.isnull(
                            row[column]) else row[column], axis=1)
                else:
                    # if the column is int, fill the missing value by the most common value of the columns with the same classify value
                    file[column] = file.apply(
                        lambda row: file[file[classify] == row[classify]][column].mode()[0] if pd.isnull(
                            row[column]) else row[column], axis=1)


def normalize(file, classify):
    """
        normalize the data
        :param file: the data
        :param classify: the classify column name
    """
    cols_to_norm =[]
    for column in file.columns:
        if column != classify:
            if pd.api.types.is_numeric_dtype(file[column]):
                # if the column is numeric, add to the columns list the need to normalize
                cols_to_norm.append(column)
    if cols_to_norm.__len__() > 0:
        # if there are columns to normalize, normalize them
        file[cols_to_norm] = StandardScaler().fit_transform(file[cols_to_norm])


def discretize(file, classify, bins, desc_type):
    """
        discretize the data by the methode given in desc_type with the number of bins given in bins
        :param file: the data
        :param classify: the classify column name
        :param bins: the number of bins
        :param desc_type: the type of discretization needed
    """
    for column in file.columns:
        if column != classify:
            if file[column].dtypes == "float64":
                # if the column is float, discretize it by the chosen methode
                if desc_type == "Equal-frequency":
                    file[column] = pd.qcut(file[column], q=bins, precision=0, labels=False)
                # or
                elif desc_type == "Equal-width":
                    file[column] = pd.cut(file[column], bins=bins, precision=0, labels=False)
                # or
                else:
                    disc = mdlp.MDLP()
                    bins = disc.cut_points(file[column].values, file[classify].values)
                    file[column] = disc.discretize_feature(file[column], bins)
                    file[column] = np.where(file[column].isnull(), np.nan, file[column])


def encodeAndPopClass(file, classify):
    """
        encode the data and pop the classify column.
        :param file: the data
        :param classify: the classify column name
        :return: the encoded data and the encoded classify column
    """
    for column in file.columns:
        file[column] = LabelEncoder().fit_transform(file[column])
    classify_col = file[classify]
    data_cols = file[file.columns[~file.columns.isin([classify])]]
    return data_cols, classify_col


def SplitTrainTest(data_columns, classify_column):
    """
        split the data into train and test.
        :param data_columns: the data (encoded)
        :param classify_column: the classify column (encoded)
        :return: the train and test data as data columns and classify column each
    """
    data_train, data_test, class_train, class_test = train_test_split(data_columns, classify_column, train_size=0.67)
    return data_train, data_test, class_train, class_test


def skDecisionTree(data_train, class_train):
    """
     builds sklearn's decision tree model
    :param data_train: the train data
    :param class_train: the train classify column
    :return: the trained model
    """
    dtc = DecisionTreeClassifier(criterion='entropy')
    dtc.fit(data_train, class_train)
    return dtc


def selfDecisionTree(data_train, class_train, data_test):
    """self decision tree"""
    # todo: self made decision tree


def skNaiveBayes(data_train, class_train):
    """
        builds sklearn's naive bayes model
        :param data_train: the train data
        :param class_train: the train classify column
        :return: the trained model
    """
    gnb = GaussianNB()
    gnb.fit(data_train, class_train)
    return gnb


class selfNaiveBayes:
    """
        class that builds self made naive bayes model and predict by new data
    """
    def __init__(self, data_train, class_train):
        """
            init the naive bayes with the train data
            :param data_train: the train data
            :param class_train: the train classify column
        """
        self.data_train = data_train
        self.class_train = class_train
        self.bayesCalcs = {}
        self.pClass = class_train.value_counts() / (len(class_train))
        self.pClass.index.name = class_train.name
        temp = data_train.join(class_train.to_frame(), how="inner")
        for column in temp.columns:
            if column != class_train.name:
                self.bayesCalcs[column] = temp.groupby([class_train.name, column]).size() / (len(temp)) / (self.pClass)

    def calcBayes(self,*args):
        """
            calculate the bayes probability
            :param args: the data (row of data from data_test)
            :return: the predicted classify
        """
        classify = ("", 0)
        calc = 1
        for classOpt in self.class_train.unique():
            calc = 1
            for column in self.data_train.columns:
                try:
                    calc *= self.bayesCalcs[column][classOpt][args[0][self.data_train.columns.get_loc(column)]]
                except KeyError:
                    calc *= 1
            calc *= self.pClass[classOpt]
            if calc > classify[1]:
                classify = (classOpt, calc)
        return classify[0]


def KNN(data_train, class_train):
    """
        builds sklearn's kNN model
        :param data_train: the train data
        :param class_train: the train classify column
        :return: the trained model
    """
    knn = KNeighborsClassifier()
    knn.fit(data_train, class_train)
    return knn


def kMeans(data_train, class_train):
    """
        builds sklearn's k-means model
        :param data_train: the train data
        :param class_train: the train classify column
        :return: the trained model
    """
    km = KMeans(n_clusters=len(class_train.unique()))
    km.fit(data_train)
    return km


def predict(model, data_train, data_test, implement):
    """
        predict the classify with the model given in model param and the data given in data_test param
        and the implement given in implement param
        :param data_train: the train data
        :param implement: kind of implement
        :param model: the trained model
        :param data_test: the test data
        :return: the prediction of the model as numpy array
    """
    if implement == "Sklearn":
        return model.predict(data_train), model.predict(data_test)
    else:
        # prediction of test data for the self-made NB model
        test_pred_array = np.array([])
        for row in data_test.values.tolist():
            # run the test prediction for each row of the given data and append it to the test_pred_array
            if test_pred_array.size == 0:
                test_pred_array = np.array([model.calcBayes(row)])
            else:
                test_pred_array = np.append(test_pred_array, model.calcBayes(row))

        # prediction of train data for the self-made NB model
        train_pred_array = np.array([])
        for row in data_train.values.tolist():
            # run the train prediction for each row of the given data and append it to the train_pred_array
            if train_pred_array.size == 0:
                train_pred_array = np.array([model.calcBayes(row)])
            else:
                train_pred_array = np.append(train_pred_array, model.calcBayes(row))
        return train_pred_array, test_pred_array


def confusionMatrix(train_pred_array, test_pred_array, class_train, class_test, model_name, axis_values):
    """
        saves the confusion matrixs of the train and test as new png file
        :param class_train: the train classify column
        :param test_pred_array: the prediction of the test data
        :param train_pred_array: the prediction of the train data
        :param axis_values: the values needed for the axis (unique values of the classify column)
        :param model_name: the name of the model used to predict
        :param class_test: the test classify column
    """
    # creating the matrix for the train prediction
    confusion_matrix = metrics.confusion_matrix(class_train, train_pred_array)
    matrix_df = pd.DataFrame(confusion_matrix)

    # all the heatmap settings
    sns.set(font_scale=1.2)
    plt.figure(figsize=(3, 2))
    ax = sns.heatmap(matrix_df, annot=True, fmt="g")
    ax.set_title("Confusion Matrix - " + model_name)
    ax.set_xlabel('Predicted classify')
    ax.set_ylabel('True classify')
    ax.xaxis.set_ticklabels(axis_values)
    ax.yaxis.set_ticklabels(axis_values)

    # add the accuracy of train to the heatmap
    ax.text(-0.5, 3.2, "Accuracy: " + str(metrics.accuracy_score(class_train, train_pred_array)))

    # save the heatmap and added text as png file
    plt.savefig('Confusion_Matrix_train.png', bbox_inches="tight", dpi=100)

    # creating the matrix for the test prediction
    confusion_matrix = metrics.confusion_matrix(class_test, test_pred_array)
    matrix_df = pd.DataFrame(confusion_matrix)

    # all the heatmap settings
    sns.set(font_scale=1.2)
    plt.figure(figsize=(3, 2))
    ax = sns.heatmap(matrix_df, annot=True, fmt="g")
    ax.set_title("Confusion Matrix - " + model_name)
    ax.set_xlabel('Predicted classify')
    ax.set_ylabel('True classify')
    ax.xaxis.set_ticklabels(axis_values)
    ax.yaxis.set_ticklabels(axis_values)

    # add the accuracy of test to the heatmap
    ax.text(-0.5, 3.2, "Accuracy: " + str(metrics.accuracy_score(class_test, test_pred_array)))

    # save the heatmap and added text as png file
    plt.savefig('Confusion_Matrix_test.png', bbox_inches="tight", dpi=100)


