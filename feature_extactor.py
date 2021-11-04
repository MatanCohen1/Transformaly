import os
import pickle
from collections import defaultdict
from os.path import join

import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from pytorch_pretrained_vit.utils import *
# from utils.evaluate_utils import contrastive_evaluate

EPS=1e-8


class FeatureExtractor():
    def __init__(self,
                 base_path,
                 model,
                 model_name,
                 verbose=1,
                 ):

        self.base_path = base_path

        # set extracted features dictionary
        self.extracted_features_path = join(self.base_path, 'extracted_features')
        if not os.path.exists(self.extracted_features_path):
            os.mkdir(self.extracted_features_path)

        # set distances dictionary
        self.features_distances = join(self.base_path, 'features_distances')
        if not os.path.exists(self.features_distances):
            os.mkdir(self.features_distances)

        # set model to eval mode
        if model is not None:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model.eval()

        self.model = model
        self.model_name = model_name
        self.verbose = verbose

    def get_clustered_features(self,
                               dataLoader,
                               early_break=-1,
                               forward_pass=None):

        """returning the model representation of the samples in dataloder"""
        clustered_features = []
        for i, (data, _) in enumerate(tqdm(dataLoader)):

            if early_break > 0 and early_break < i:
                break

            if forward_pass is not None:
                encoded_outputs = self.model(data.to('cuda'), forward_pass=forward_pass)
            else:
                encoded_outputs = self.model(data.to('cuda'))

            if isinstance(encoded_outputs, list) and len(encoded_outputs) == 1:
                encoded_outputs = encoded_outputs[0]  # just one head, to check after!!!!!
            clustered_features.append(encoded_outputs.detach().cpu().numpy())

        clustered_features = np.concatenate(clustered_features)
        return clustered_features

    def extract_features(self,
                         trainsetLoader,
                         testsetLoader
                         ):

        train_features = self.get_clustered_features(trainsetLoader)
        test_features = self.get_clustered_features(testsetLoader)

        return train_features, test_features

    def save_extracted_features(self,
                                train_features,
                                test_features,
                                use_softmax):

        softmax_str = 'softmax_' if use_softmax else ''
        with open(join(self.extracted_features_path,
                       f'{softmax_str}train_{self.model_name}_features.npy'), 'wb') as f:
            np.save(f, train_features)

        with open(join(self.extracted_features_path,
                       f'{softmax_str}test_{self.model_name}_features.npy'), 'wb') as f:
            np.save(f, test_features)

    def return_train_features_path(self,
                                   use_softmax
                                   ):
        softmax_str = 'softmax_' if use_softmax else ''
        return join(self.extracted_features_path,
                    f'{softmax_str}train_{self.model_name}_features.npy')

    def load_extracted_features(self,
                                use_softmax):
        softmax_str = 'softmax_' if use_softmax else ''

        with open(join(self.extracted_features_path,
                       f'{softmax_str}train_{self.model_name}_features.npy'), 'rb') as f:
            train_features = np.load(f)

        with open(join(self.extracted_features_path,
                       f'{softmax_str}test_{self.model_name}_features.npy'), 'rb') as f:
            test_features = np.load(f)

        if use_softmax:
            train_features = softmax(train_features, axis=1)
            test_features = softmax(test_features, axis=1)

        return train_features, test_features

    def get_sample_neighborhood_inds(self,
                                     train_features,
                                     test_feature,
                                     k,
                                     return_dists=False,
                                     sort=True):
        """return the test_feature k nearest train_features neighbors indexs"""
        test_feature_dist = train_features - test_feature
        test_feature_dist = np.sqrt(np.sum(test_feature_dist ** 2, 1))
        k_closest_inds = test_feature_dist.argsort()[:k]

        if return_dists:
            test_feature_dist = np.sort(test_feature_dist)
            return k_closest_inds, test_feature_dist[:k]

        return k_closest_inds

    def get_features_info(self,
                          train_features,
                          test_features,
                          k,
                          remove_first=False
                          ):

        # get train samples distances and neighbors indexs
        train_features = np.array(train_features)
        train_samples_info = defaultdict(dict)

        for i, test_feature in enumerate(tqdm(test_features)):

            # calculate the regular sample neighborhood's samples knn distance
            sample_neighborhood_inds, train_sample_feature_dist = self.get_sample_neighborhood_inds(
                train_features,
                test_feature,
                k + 1 if remove_first else k,
                return_dists=True)

            # remove the distance of tha sample from itelf( zero val)
            if remove_first:
                train_sample_feature_dist = np.array(train_sample_feature_dist[1:])
                sample_neighborhood_inds = np.array(sample_neighborhood_inds[1:])

            # calculate the regular knn distance
            train_samples_info[i]['sorted_distances_from_train_samples'] = train_sample_feature_dist
            train_samples_info[i]['mean_sample_distances'] = np.mean(train_sample_feature_dist)
            train_samples_info[i]['sample_neighborhood_inds'] = sample_neighborhood_inds

        # sort the distance dictionary by mean k distance
        train_samples_info = dict(sorted(train_samples_info.items(),
                                         key=lambda item: item[1]['mean_sample_distances']))

        return train_samples_info

    def get_features_dists(self,
                           train_features,
                           k):
        # get train distances vectors
        train_features_dists = []
        for train_feature in tqdm(train_features):
            _, train_sample_feature_dist = self.get_sample_neighborhood_inds(train_features,
                                                                             train_feature,
                                                                             k + 1,
                                                                             return_dists=True)

            # remove the distance of tha sample from itelf( zero val)
            train_sample_feature_dist = np.array(train_sample_feature_dist[1:])
            train_features_dists.append(train_sample_feature_dist)
        train_features_dists = np.array(train_features_dists)
        return train_features_dists

    def save_features_info(self,
                           train_samples_info,
                           test_samples_info,
                           train_features_dists,
                           use_softmax
                           ):

        softmax_str = 'softmax_' if use_softmax else ''
        with open(join(self.features_distances, f'{softmax_str}train_samples_info.pkl'), 'wb') as f:
            pickle.dump(train_samples_info, f, pickle.HIGHEST_PROTOCOL)

        with open(join(self.features_distances, f'{softmax_str}test_samples_info.pkl'), 'wb') as f:
            pickle.dump(test_samples_info, f, pickle.HIGHEST_PROTOCOL)

        with open(join(self.features_distances, f'{softmax_str}train_features_dists.pkl'), 'wb') as f:
            pickle.dump(train_features_dists, f, pickle.HIGHEST_PROTOCOL)

    def calculate_distances(self,
                            train_features,
                            test_features,
                            k
                            ):

        # calculate features distances

        train_samples_info = self.get_features_info(train_features=train_features,
                                                    test_features=train_features,
                                                    k=k,
                                                    remove_first=True)

        test_samples_info = self.get_features_info(train_features=train_features,
                                                   test_features=test_features,
                                                   k=k,
                                                   remove_first=False)

        train_features_dists = self.get_features_dists(train_features,
                                                       k=k)

        return train_samples_info, test_samples_info, train_features_dists


    def _get_features(self,
                      trainsetLoader,
                      testsetLoader,
                      use_softmax=True
                      ):
        softmax_str = 'softmax_' if use_softmax else ''

        # extract features and save it
        train_file_path = join(self.extracted_features_path,
                               f'{softmax_str}train_{self.model_name}_features.npy')

        test_file_path = join(self.extracted_features_path,
                              f'{softmax_str}test_{self.model_name}_features.npy')

        if not (os.path.exists(train_file_path) and os.path.exists(test_file_path)):
            if self.verbose:
                print("================================================================")
                print("======================= Extract Features =======================")

            train_features, test_features = self.extract_features(trainsetLoader=trainsetLoader,
                                                                  testsetLoader=testsetLoader)

            if self.verbose:
                print("======================== Save Features =========================")
            self.save_extracted_features(train_features=train_features,
                                         test_features=test_features,
                                         use_softmax=use_softmax)
        else:
            print("------ train_features and test_features already exists ------")
            print(f"------ load them from {self.extracted_features_path} ------")
            with open(train_file_path, 'rb') as f:
                train_features = np.load(f)

            with open(test_file_path, 'rb') as f:
                test_features = np.load(f)
        return train_features, test_features

    def _get_samples_info(self,
                          train_features,
                          test_features,
                          k,
                          use_softmax=True):

        softmax_str = 'softmax_' if use_softmax else ''
        # extract features and save it
        train_file_path = join(self.features_distances,
                               f'{softmax_str}train_samples_info.pkl')

        test_file_path = join(self.features_distances,
                              f'{softmax_str}test_samples_info.pkl')

        if not (os.path.exists(train_file_path) and os.path.exists(test_file_path)):

            if self.verbose:
                print("================= Extract Train And Test Infos  ================")
            train_samples_info, test_samples_info, train_features_dists = self.calculate_distances(
                train_features=train_features,
                test_features=test_features,
                k=k)

            if self.verbose:
                print("=================== Save Train And Test Infos ==================")
            self.save_features_info(train_samples_info=train_samples_info,
                                    test_samples_info=test_samples_info,
                                    train_features_dists=train_features_dists,
                                    use_softmax=use_softmax
                                    )

            if self.verbose:
                print("================================================================")

        else:
            print(
                f"------ train_samples_info and test_samples_info already exists here {self.features_distances}------")

    def scoring_preprsocessing(self,
                               trainsetLoader,
                               testsetLoader,
                               k,
                               use_softmax=True
                               ):

        train_features, test_features = self._get_features(trainsetLoader,
                                                           testsetLoader,
                                                           use_softmax=use_softmax)

        if use_softmax:
            train_features = softmax(train_features, axis=1)
            test_features = softmax(test_features, axis=1)

        self._get_samples_info(train_features=train_features,
                               test_features=test_features,
                               k=k,
                               use_softmax=use_softmax)









def scoring_preprocessing(base_feature_path,
                          dataset,
                          anomaly_classes,
                          train_transforms,
                          val_transforms,
                          use_coarse_labels,
                          model,
                          model_name,
                          k_max=1200,
                          batch_size = 256,
                          verbose=0

                          ):
    trainsetLoader, testsetLoader = get_train_and_test_dataloaders(dataset=dataset,
                                                                   anomaly_classes=anomaly_classes,
                                                                   train_transforms=train_transforms,
                                                                   val_transforms=val_transforms,
                                                                   use_coarse_labels=use_coarse_labels,
                                                                   batch_size=batch_size
                                                                   )
    fe = FeatureExtractor(base_path=base_feature_path,
                          model=model,
                          model_name=model_name,
                          verbose=verbose
                          )

    fe.scoring_preprsocessing(trainsetLoader=trainsetLoader,
                              testsetLoader=testsetLoader,
                              k=k_max,
                              use_softmax=False,
                              )

    #TODO: UNCOMMIT IT

    # fe.scoring_preprsocessing(trainsetLoader=trainsetLoader,
    #                           testsetLoader=testsetLoader,
    #                           k=k_max,
    #                           use_softmax=True,
    #                           )
    return fe
