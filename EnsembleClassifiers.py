"""

    Course: Supervised and Experienced Learning
    Professor: Miquel Sanchez i Marre
    Title: Random Forest, An Ensemble Classifier
    Description: An ensemble classfier-regressor, featuring a random selection of features to split on at each node.
                 Building a large number of un-pruned decision trees. The motivation was minimizing error correlation
                 among classifiers in the ensemble.
    Author: Pablo Eliseo Reynoso Aguirre
    Submission: December 14, 2017.



    Class: EnsembleClassifiers.py, aims to ease the randomforest and CART decision trees (gini-index) usage.

"""

from itertools import chain;
from collections import Counter;

import random as rndm;
import CART;


class EnsembleClassifiers():

    n_features = 0;
    n_trees = 0;

    cart = CART.CART();
    trees_features = [];

    def retrieve_forest_feature_relevance(self, features_lists):

        random_forest_features = list(chain.from_iterable(features_lists));
        self.trees_features = Counter(random_forest_features);


    def sample_replacement(self, dataset, ratio):

        rndm.seed(9);

        sample = [];
        n_sample = round(len(dataset) * ratio);

        while len(sample) < n_sample:
            index = rndm.randrange(len(dataset));
            sample.append(dataset[index]);

        return sample;



    def predict(self, trees, sample, regression=False):

        h_x = [self.cart.predict(tree, sample) for tree in trees];
        if regression:
            return sum(h_x)/len(h_x);
        else:
            return max(set(h_x), key=h_x.count);



    def random_forest(self, train_set, test_set, regression, max_depth, min_size, sample_size, n_trees, n_features):

        trees = [];
        features_lists = [];
        for t in range(n_trees):

            sample = self.sample_replacement(train_set, sample_size);
            tree = self.cart.build_tree(sample, max_depth, min_size, n_features);
            features_lists.append(self.cart.retrieve_features(tree));
            trees.append(tree);

        self.retrieve_forest_feature_relevance(features_lists);
        H_x = [self.predict(trees, sample, regression) for sample in test_set];
        return (H_x);



