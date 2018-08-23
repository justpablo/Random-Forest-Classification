"""

    Course: Supervised and Experienced Learning
    Professor: Miquel Sanchez i Marre
    Title: Random Forest, An Ensemble Classifier
    Description: An ensemble classfier-regressor, featuring a random selection of features to split on at each node.
                 Building a large number of un-pruned decision trees. The motivation was minimizing error correlation
                 among classifiers in the ensemble.
    Author: Pablo Eliseo Reynoso Aguirre
    Submission: December 14, 2017.



    Class: DataPreprocessing.py, aims to ease the preprocess of the databases, cleaning, feature conversion, slicing.

"""

import pandas as pd;
import numpy as np;
import re;

from sklearn.feature_extraction.text import TfidfVectorizer;
from nltk.corpus import stopwords;


class DataPreprocessing():


    n_samples = 0;
    n_features = 0;

    csv_datasets = ["./Data/contact_lenses.csv",
                    "./Data/zoo_animal_data/zoo.csv",
                    "./Data/sms_spam_data/sms_spam_processed.csv",
                    "./Data/mushrooms.csv",
                    "./Data/taekwondo_techniques_data/taekwondo_clean_s.csv",
                    "./Data/mlb_baseball.csv",
                    ];

    csv_datasets_col_names = [['age', 'visual_deficiency', 'astigmatism', 'production', 'lents'],

                              ['name','hair','feathers','eggs','milk','airborne','aquatic','predator',
                               'toothed','backbone','breathes','venomous','fins','legs','tail','domestic',
                               'catsize','class_type'],

                              ['call', 'free', 'get', 'go', 'gt', 'know', 'lt', 'lt gt', 'ok', 'ur', 'class'],

                              ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
                               'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape',
                               'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
                               'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color',
                               'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat','class'],

                              ['id','sex','age','weight','experience','technique','trial','measuring','belt'],

                              ['rank', 'year', 'team', 'league', 'games_played', 'losses', 'ties',
                               'win_lost_percent', 'pythagorean_win_lost', 'finish', 'games_behind', 'playoffs',
                               'runs_scored','runs_allowed', 'attendance', 'bats_avg_age', 'pit_avg_age', 'n_players',
                               'n_pitchers', 'top_player','managers','current','wins']

                              ];


    features_reference_dictionary = [];

    not_continouos_attrs = False;


    def merge_taekwondo_datasets(self):


        data_raw = pd.read_csv("./Data/taekwondo_techniques_data/taekwondo_feats1.csv", header=None);
        data_raw = data_raw.as_matrix();

        fd = open('./Data/taekwondo_techniques_data/taekwondo.csv', 'wb');
        fd.write("id,technique,trial,measuring");
        for i in range(1, len(data_raw[0, :])):

            technique = data_raw[0, i];
            id = data_raw[1, i];
            trial = data_raw[2, i];

            for j in range(3, len(data_raw[:, i])):

                if data_raw[j, i] is np.NaN:
                    break;
                fd.write("\n" + str(id) + "," + str(technique) + "," + str(trial) + "," + str(data_raw[j, i]));

        fd.close();


        taekwondo_dataset = [];

        id_reference = pd.read_csv("./Data/taekwondo_techniques_data/taekwondo_feats2.csv");
        id_reference = id_reference.as_matrix();

        taekw_clean = pd.read_csv("./Data/taekwondo_techniques_data/taekwondo.csv");
        taekw_clean = taekw_clean.as_matrix();

        for i in range(taekw_clean.shape[0]):

            id_feats = id_reference[(id_reference[:,0] == taekw_clean[i,0])];
            taekwondo_dataset.append([taekw_clean[i,0],
                                      id_feats[0,1],
                                      id_feats[0,2],
                                      id_feats[0,3],
                                      id_feats[0,4],
                                      taekw_clean[i,1],
                                      taekw_clean[i,2],
                                      taekw_clean[i,3],
                                      id_feats[0, 5],]
                                     );

        fd = open('./Data/taekwondo_techniques_data/taekwondo_clean.csv', 'wb');
        fd.write("id,sex,age,weight,experience,technique,trial,measuring,belt");

        for sample in taekwondo_dataset:

            row = ','.join(map(str, sample));
            fd.write('\n'+row);

        fd.close();

        data_ = pd.read_csv('./Data/taekwondo_techniques_data/taekwondo_clean.csv');
        data_.columns = self.csv_datasets_col_names[4];
        data_shuffled = data_.sample(frac=1).reset_index(drop=True);
        data_shuffled.to_csv('./Data/taekwondo_techniques_data/taekwondo_clean_s.csv', sep='\t', encoding='utf-8', index=False);


    def preprocess_sms_dataset(self):

        sms_raw = pd.read_csv("./Data/sms_spam_data/sms_spam.csv");
        sms_raw = sms_raw.as_matrix();

        sms_labels = [];
        sms_messages = [];

        for i in range(sms_raw.shape[0]):

            sms_labels.append(sms_raw[i,0]);
            sms_messages.append(re.sub(r'\W+', ' ', sms_raw[i,1]).lower());


        stopwords_list = set(stopwords.words('english'));
        vect = TfidfVectorizer(stop_words=stopwords_list,
                               max_features=10,
                               max_df=0.35,
                               ngram_range=(1,3),
                               strip_accents='unicode');

        sms_features = vect.fit_transform(sms_messages).toarray();
        dtm_vocabulary = vect.get_feature_names();

        fd = open('./Data/sms_spam_data/sms_spam_processed.csv', 'wb');
        for i in range(len(dtm_vocabulary)):
            fd.write(dtm_vocabulary[i].encode('utf-8')+',');
        fd.write('CLASS');

        for i in range(sms_features.shape[0]):
            row = ','.join(map(str, sms_features[i,:]));
            fd.write('\n' + row +','+ sms_labels[i]);

        fd.close();



    def csv_processor(self, csv_path, feature_names):

        dataset = pd.read_csv(csv_path);
        dataset.columns = feature_names;
        return dataset;



    def fix_dataset_missing_values(self, dataset):

        for column in dataset.columns:
            dataset[column] = dataset[column].replace('?', np.NaN);
            dataset[column] = dataset[column].replace('--', 0);
            dataset[column] = dataset[column].fillna(dataset[column].value_counts().index[0]);



    def repair_dataset_attributes(self, dataset, features):

        self.fix_dataset_missing_values(dataset);

        self.n_samples = dataset.shape[0];
        self.n_features = dataset.shape[1] - 1;

        if self.not_continouos_attrs is True:

            for feat in features:
                if dataset[feat].dtype == np.float64:
                    dataset[
                        feat] *= 10;  #multiplying by 10 to constrain attr to 10 categories
                    dataset[feat] = dataset[feat].astype(int);



    def convert_categorical_to_numerical_attrs(self, dataset):

        for column in dataset.columns:

            if dataset[column].dtype == np.object:
                attr_vals = dataset[column].unique();

                if len(attr_vals) > 1000: continue;
                lookup = dict();

                for i, value in enumerate(attr_vals):
                    lookup[value] = i;

                dataset[column].replace(lookup, inplace=True);

                self.features_reference_dictionary.append((column,lookup));



    def display_data_info(self, dataset):

        print("\n1. Number of samples: " + str(self.n_samples));
        print("\n2. Number of features: " + str(self.n_features));
        print("\n3. Feature types:");
        print(dataset.dtypes);



    def run_preprocessing(self, selector):

        self.features_reference_dictionary = [];

        print('A) ::Processing CSV files::');
        dataset = self.csv_processor(self.csv_datasets[selector], self.csv_datasets_col_names[selector]);

        print('B) ::Casting continuous to integers and repairing NANs attributes in Dataset::');
        self.repair_dataset_attributes(dataset, dataset.columns);

        print('C) ::Converting categorical-numerical attributes in Dataset::');
        self.convert_categorical_to_numerical_attrs(dataset);

        print('D) ::Display Mapping of Categorical Attributes to Numerical Lookups');
        print(self.features_reference_dictionary);

        return dataset;


