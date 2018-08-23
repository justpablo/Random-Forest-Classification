"""

    Course: Supervised and Experienced Learning
    Professor: Miquel Sanchez i Marre
    Title: Random Forest, An Ensemble Classifier
    Description: An ensemble classfier-regressor, featuring a random selection of features to split on at each node.
                 Building a large number of un-pruned decision trees. The motivation was minimizing error correlation
                 among classifiers in the ensemble.
    Author: Pablo Eliseo Reynoso Aguirre
    Submission: December 14, 2017.



    Class: ModelValidation.py, aims to ease the ensemble classifier performance evaluation by k-fold cross validation
"""


import random as rndm;
import numpy as np;

class ModelValidation():


    def dataset_cv_split(self, dataset, k_folds):

        dataset_cv = [];
        dataset_ = np.copy(dataset);

        fold_size = int(dataset_.shape[0]/k_folds);
        folds_sizes = [fold_size] * k_folds;

        leftover = dataset_.shape[0] - fold_size*k_folds;

        if leftover > 0:

            for i in range(len(folds_sizes)):
                folds_sizes[i] += 1;
                leftover -= 1;

                if leftover == 0:
                    break;

        for i in range(k_folds):

            fold = [];
            while len(fold) < folds_sizes[i]:

                index = rndm.randrange(dataset_.shape[0]);
                fold.append(dataset_[index,:]);
                dataset_ = np.delete(dataset_, (index), axis=0);

            dataset_cv.append(fold);

        return dataset_cv;



    def accuracy(self, y, h_x):

        hits = 0;
        for i in range(len(y)):

            if y[i] == h_x[i]:
                hits += 1;
        return hits / float(len(y)) * 100.0;




    def mean_absolute_percentage_error(self, y_true, y_pred):

        abs_perc_err = [];
        for y, h in zip(y_true, y_pred):
            abs_perc_err.append(np.mean(np.abs((y-h)/y)));

        return sum(abs_perc_err)/len(abs_perc_err)*100;




    def remove_fold(self, fold, train_set):

        fold_i = -1;
        for i in range(len(train_set)):
            if np.array_equal(train_set[i], fold):
                fold_i = i;
                break;

        if fold_i > -1:
            train_set.pop(fold_i);
        else:
            raise ValueError('Error: fold not found in train_set.');




    def evaluate_algorithm(self, dataset, algorithm, k_folds, regression, *args):

        print("MV:evaluate_algorithm()");

        folds = self.dataset_cv_split(dataset, k_folds);

        scores = [];

        for fold in folds:

            train_set = list(folds);
            self.remove_fold(fold,train_set);
            train_set = sum(train_set, []);

            test_set = [];
            for sample in fold:

                sample_ = list(sample);
                sample_[-1] = None;
                test_set.append(sample_);



            h_x = algorithm(train_set, test_set, regression, *args);
            y = [sample[-1] for sample in fold];
            if regression:
                outcome = self.mean_absolute_percentage_error(y,h_x);
                print("MAPE: " + str(outcome));
            else:
                outcome = self.accuracy(y, h_x);
                print("accuracy: " + str(outcome));
            scores.append(outcome);


        return scores;
