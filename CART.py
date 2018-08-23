"""

    Course: Supervised and Experienced Learning
    Professor: Miquel Sanchez i Marre
    Title: Random Forest, An Ensemble Classifier
    Description: An ensemble classfier-regressor, featuring a random selection of features to split on at each node.
                 Building a large number of un-pruned decision trees. The motivation was minimizing error correlation
                 among classifiers in the ensemble.
    Author: Pablo Eliseo Reynoso Aguirre
    Submission: December 14, 2017.



    Class: CART.py, intends to ease the CART decision trees (gini-index) usage. An Gini Impurity Based Decision Tree
                    that aims to find the the most pure binary splits in every node of the tree in terms of classes
                    mixture at both sides of the split leading to a natural discriminant decision by building
                    a non-complex tree topology.

"""

import numpy as np;
import random as rndm;

class CART():


    def attribute_binary_split(self, attr, value, samples):

        left, right = [], [];
        for sample in samples:

            if sample[attr] < value:
                left.append(sample);
            else:
                right.append(sample);

        return left, right;



    def gini_index(self, attr_split, labels):

        n_instances = float(sum([len(split) for split in attr_split]));

        gini = 0.0;
        for split in attr_split:
            size = float(len(split));

            if size == 0:
                continue;

            score = 0.0;
            for label in labels:
                p = [row[-1] for row in split].count(label) / size;
                score += np.power(p,2);

            gini += (1.0 - score) * (size / n_instances);

        return gini



    def find_best_split(self, samples, n_features):

        rndm.seed(2);

        labels = list(set(sample[-1] for sample in samples));

        n_attr, n_value, n_gini, n_splits = -1, -1, 1, None;
        features = [];

        while len(features) < n_features:

            attr = rndm.randrange(len(samples[0]) - 1);
            if attr not in features:
                features.append(attr);

        for attr in features:
            for sample in samples:

                attr_splits = self.attribute_binary_split(attr, sample[attr], samples);
                gini = self.gini_index(attr_splits, labels);

                if gini < n_gini:
                    n_attr, n_value, n_gini, n_splits = attr, sample[attr], gini, attr_splits;

        return {'feature': n_attr, 'value': n_value, 'branches': n_splits};



    def create_leave(self, branch):

        labels = [row[-1] for row in branch];
        return max(set(labels), key=labels.count);



    def expand_tree(self, node, max_depth, min_size, n_features, depth):

        left, right = node['branches'];
        del (node['branches']);

        # merging branches
        if not left or not right:
            node['left'] = node['right'] = self.create_leave(left + right);
            return;

        # bounding max_depth
        if depth >= max_depth:
            node['left'] = self.create_leave(left);
            node['right'] = self.create_leave(right);
            return;

        # bounding min_size / expand in left
        if len(left) <= min_size:
            node['left'] = self.create_leave(left);
        else:
            node['left'] = self.find_best_split(left, n_features);
            self.expand_tree(node['left'], max_depth, min_size, n_features, depth + 1);

        # bounding min_size / expand in right
        if len(right) <= min_size:
            node['right'] = self.create_leave(right);
        else:
            node['right'] = self.find_best_split(right, n_features);
            self.expand_tree(node['right'], max_depth, min_size, n_features, depth + 1);



    def build_tree(self, train, max_depth, min_size, n_features):

        root = self.find_best_split(train, n_features);
        self.expand_tree(root, max_depth, min_size, n_features, 1);
        return root;



    def predict(self, node, sample):

        if sample[node['feature']] < node['value']:

            if isinstance(node['left'], dict):
                return self.predict(node['left'], sample);
            else:
                return node['left'];

        else:

            if isinstance(node['right'], dict):
                return self.predict(node['right'], sample);
            else:
                return node['right'];



    def retrieve_features(self, tree):
        return set(list(self.find_features(tree)));



    def find_features(self, tree, key="feature"):

        for k, v in tree.iteritems():

            if k == key:
                yield v;

            elif isinstance(v, dict):
                for result in self.find_features(v,key):
                    yield result;

            elif isinstance(v, list):
                for d in v:
                    for result in self.find_features(d,key):
                        yield result;



    def display_tree(self, tree, depth=0):

        for k,v in sorted(tree.items(),key=lambda x: x[0]):

            if isinstance(v, dict):
                print(("  ")*depth + ("%s" % k));
                self.display_tree(v,depth+1);

            else:
                print(("  ")*depth + "%s %s" % (k, v));



