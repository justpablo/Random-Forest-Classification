"""

    Course: Supervised and Experienced Learning
    Professor: Miquel Sanchez i Marre
    Title: Random Forest, An Ensemble Classifier
    Description: An ensemble classfier-regressor, featuring a random selection of features to split on at each node.
                 Building a large number of un-pruned decision trees. The motivation was minimizing error correlation
                 among classifiers in the ensemble.
    Author: Pablo Eliseo Reynoso Aguirre
    Submission: December 14, 2017.



    Class: __main__.py, aims to start the general data analysis.

    :::Datasets:::

    0. Contact Lenses
    1. Zoo
    2. SMS Spam
    3. Mushrooms Poisoned?
    4. Taekwondo
    5. MLB 1870-2016



"""

import numpy as np;
import math;
import time;

import DataPreprocessing;
import EnsembleClassifiers;
import ModelValidation;

DP = DataPreprocessing.DataPreprocessing();
EC = EnsembleClassifiers.EnsembleClassifiers();
MV = ModelValidation.ModelValidation();


#Special Cases Pre-processing

#DP.merge_taekwondo_datasets();
#DP.preprocess_sms_dataset();


def dataset_learning(dataset,output_file,dataset_name,preprocess_time,NT,F,parameters):

    txtfile = open('./Learning_Results/'+output_file+'.txt','wb');
    txtfile.write("\n::::::::::::::::::::::::::::");
    txtfile.write("\nRandom Forest Classification");
    txtfile.write("\n\n********************************");
    txtfile.write("\nDataset Name: "+dataset_name);
    txtfile.write("\nPre-processing Time: "+str(preprocess_time)+" s");
    txtfile.write("\nLearning Configurations: ");

    for j in range(len(NT)):

        print("range_j:: "+str(j));

        for k in range(len(F)):

            print("range_k:: " + str(k));

            learning_t0 = time.time();
            scores = MV.evaluate_algorithm(dataset,EC.random_forest,parameters[0],parameters[1],parameters[2],
                                           parameters[3],parameters[4],NT[j],F[k]);
            learning_tf = time.time();


            txtfile.write("\n\n------------------------------");
            txtfile.write("\nNumber of Trees: " + str(NT[j]));
            txtfile.write("\nNumber of Features: " + str(F[k]));
            txtfile.write("\nTree Max Depth: " + str(tree_max_depth));
            txtfile.write("\nNode Min Elements: " + str(node_min_elements));
            txtfile.write("\nDataset Usage Percent: " + str(train_size*100));
            txtfile.write("\nNumber of Folds: " + str(k_folds));
            txtfile.write("\nAccuracy Fold-Scores: ");
            txtfile.write(','.join(map(str, scores)));
            txtfile.write("\nOverall Accuracy: "+str(sum(scores)/float(len(scores))));
            txtfile.write("\nLearning Time: " + str(learning_tf - learning_t0) + " s");
            txtfile.write("\nRandom Forest features relevance: ");
            txtfile.write(str(EC.trees_features));

    txtfile.close();






#::::Data - Preprocessing::::
datasets_names = ["Contact Lenses","Zoo", "SMS Spam", "Mushrooms", "Taekwondo", "MLB 1870-2016"];
file_names = ["contact_lenses_learning","zoo_learning","sms_spam_learning","mushrooms_learning","taekwondo_learning","mlb_learning"];
regression_task = [False, False, False, False, False, True];


#::::Ensemble - Classifiers::::
NT = [50,100];
tree_max_depth = 10;
node_min_elements = 1;
train_size = 0.2;
regression_task1 = True;


#::::Model - Validation::::
k_folds = 5;



for d in range(len(datasets_names)):

    preprocess_t0 = time.time();
    dataset = DP.run_preprocessing(d);
    preprocess_tf = time.time();

    F = [1,3,int(math.log(DP.n_features,2)+1),int(np.sqrt(DP.n_features))];
    dataset_cat_dicitonaries = DP.features_reference_dictionary;

    parameters = [k_folds,regression_task[d],tree_max_depth,node_min_elements,train_size];
    dataset_learning(dataset,file_names[d],datasets_names[d],(preprocess_tf-preprocess_t0),NT,F,parameters);


#tqdm

