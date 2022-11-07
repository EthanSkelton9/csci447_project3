from functools import reduce as rd
import pandas as pd
import os
from itertools import product as prod
from functools import partial as pf
import time
from neuralNet import Neural_Net

class CrossValidation:
    def __init__(self, data):
        self.data = data
        self.nn = Neural_Net(data)

    '''
    @param k: the number of folds
    @param: df: an optional subdataframe
    @return: a list of lists that represents a partition of the data's index
    '''
    def stratified_partition(self, k, df = None):
        if df is None: df = self.data.df
        p = [[] for i in range(k)]
        if self.data.classification:
            def class_partition(classdf, p, c):
                n = classdf.shape[0]
                (q, r) = (n // k, n % k)
                j = 0
                for i in range(k):
                    z = (i + c) % k
                    p[z] = p[z] + [classdf.at[x, 'index'] for x in range(j, j + q + int(i < r))]
                    j += q + int(i < r)
                return (p, c + r)
            c = 0
            for cl in self.data.classes:
                classdf = df[df['Target'] == cl].reset_index()
                (p, c) = class_partition(classdf, p, c)
        else:
            sorted_df = df.sort_values(by=['Target']).reset_index()
            n = sorted_df.shape[0]
            (q, r) = (n // k, n % k)
            for i in range(k):
                p[i] = p[i] + [sorted_df.at[i + c * k, 'index'] for c in range(q + int(i < r))]
        return p

    '''
    @param df: the dataframe
    @param partition: the partition we are using
    @return: a tuple a dictionary of the training sets and a dictionary of the test sets
    '''
    def training_test_dicts(self, df, partition=None):
        if partition is None: partition = self.stratified_partition(10)
        train_dict = {}
        test_dict = {}
        for i in range(len(partition)):
            train_index = rd(lambda l1, l2: l1 + l2, partition[:i] + partition[i+1:])
            train_dict[i] = df.filter(items=train_index, axis=0)
            test_dict[i] = df.filter(items=partition[i], axis=0)
        return (train_dict, test_dict)

    '''
    @param train_dict: dictionary of the training sets
    @param eta_space: the range of learning rates
    @param alpha_space: the range of momentum weights
    @param appendCount: the number of rows to append to the dataframe
    @param csv: the csv file we are appending to
    @return: a data frame that has an error for each hyperparameter and fold combination
    '''
    def getErrorDf(self, train_dict, eta_space, alpha_space, appendCount = None, csv = None):
        def error(i):
            (f, eta, alpha) = my_space[i]
            pred = NNs[f](eta, [8, 4], alpha)
            return self.nn.calc_error(pred, target, self.data)

        start_time = time.time()
        target = self.nn.targetvec(self.data.classification, self.data.classes)
        folds = pd.Index(range(10))
        NNs = folds.map(lambda f: self.nn.stochastic_online_gd(train_dict[f]))
        my_space = pd.Series(prod(folds, eta_space, alpha_space))
        df_size = len(my_space)
        if csv is None:
            cols = list(zip(*my_space))
            col_titles = ["Fold", "eta", "alpha"]
            data = zip(col_titles, cols)
            error_df = pd.DataFrame(index=range(len(my_space)))
            for (title, col) in data:
                error_df[title] = col
            error_df["Error"] = df_size * [None]
            start = 0
            print("Table Created")
        else:
            error_df = pd.read_csv(csv, index_col=0)
            filtered = error_df["Error"].loc[pd.isnull(error_df["Error"])]
            start = filtered.index[0]
        end = df_size if appendCount is None else min(start + appendCount, df_size)
        error_df["Error"][start:end] = pd.Series(range(start, end)).map(error).values
        error_df.to_csv(os.getcwd() + '\\' + str(self.data) + '\\' + "{}_Error.csv".format(str(self.data)))
        error_df.to_csv(os.getcwd() + '\\' + str(self.data) + '\\' + "{}_Error_From_{}_To_{}.csv".format(str(self.data), start, end))
        print("Time Elapsed: {} Seconds".format(time.time() - start_time))
        return error_df

    def test(self, eta_space, alpha_space,  new = False, appendCount = None):
        p = self.stratified_partition(10)
        (train_dict, test_dict) = self.training_test_dicts(self.data.df, p)
        csv = None if new else os.getcwd() + '\\' + str(self) + '\\' + "{}_Error.csv".format(str(self.data))
        error_df = self.getErrorDf(train_dict, eta_space, alpha_space, appendCount, csv)

