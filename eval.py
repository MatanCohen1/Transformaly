"""
Transformaly Evaluation Script
"""
import os
import argparse
import logging
from os.path import join
import pandas as pd
import numpy as np
from numpy.linalg import eig
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import mixture
import torch.nn
from utils import print_and_add_to_log, get_datasets_for_ViT, \
    Identity, get_finetuned_features
from pytorch_pretrained_vit.model import AnomalyViT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--data_path', default='./data/', help='Path to the dataset')
    parser.add_argument('--whitening_threshold', default=0.9, type=float,
                        help='Explained variance of the whitening process')
    parser.add_argument('--unimodal', default=False, action='store_true',
                        help='Use the unimodal settings')
    parser.add_argument('--batch_size', type=int, default=6, help='Training batch size')
    parser_args = parser.parse_args()
    args = vars(parser_args)

    args['use_layer_outputs'] = list(range(2, 12))
    args['use_imagenet'] = True
    BASE_PATH = 'experiments'

    if args['dataset'] == 'cifar10':
        _classes = range(10)
    elif args['dataset'] == 'fmnist':
        _classes = range(10)
    elif args['dataset'] == 'cifar100':
        _classes = range(20)
    elif args['dataset'] == 'cats_vs_dogs':
        _classes = range(2)
    elif args['dataset'] == 'dior':
        _classes = range(19)
    else:
        raise ValueError(f"Does not support the {args['dataset']} dataset")

    # create the relevant directories
    if not os.path.exists(
            join(BASE_PATH,
                 f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}')):
        os.makedirs(join(BASE_PATH,
                         f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}'))

    logging.basicConfig(
        filename=join(BASE_PATH,
                      f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}',
                      f'Eval_{args["dataset"]}_Transformaly_outputs.log'), level=logging.DEBUG)

    print_and_add_to_log("========================================================",
                         logging)
    print_and_add_to_log("Args are:", logging)
    print_and_add_to_log(args, logging)
    print_and_add_to_log("========================================================",
                         logging)
    results = {'class': [],
               'pretrained_AUROC_scores': [],
               'all_layers_finetuned_AUROC_scores': [],
               'pretrained_and_finetuned_AUROC_scores': []}

    for _class in _classes:
        print_and_add_to_log("===================================", logging)
        print_and_add_to_log(f"Class is : {_class}", logging)
        print_and_add_to_log("===================================", logging)
        args['_class'] = _class
        base_feature_path = join(
            BASE_PATH,
            f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}/class_{_class}')
        model_path = join(base_feature_path, 'model')

        args['base_feature_path'] = base_feature_path
        args['model_path'] = model_path

        # create the relevant directories
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if args['unimodal']:
            anomaly_classes = [i for i in _classes if i != args['_class']]
        else:
            anomaly_classes = [args['_class']]

        results['class'].append(args['_class'])

        trainset, testset = get_datasets_for_ViT(dataset=args['dataset'],
                                                 data_path=args['data_path'],
                                                 one_vs_rest=args['unimodal'],
                                                 _class=args['_class'],
                                                 normal_test_sample_only=False,
                                                 use_imagenet=args['use_imagenet'],
                                                 )

        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=args['batch_size'],
                                                  shuffle=False)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=args['batch_size'],
                                                   shuffle=False)

        anomaly_targets = [0 if i in anomaly_classes else 1 for i in testset.targets]

        print_and_add_to_log("=====================================================",
                             logging)

        # get ViT features
        with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                       f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                       'train_pretrained_ViT_features.npy'), 'rb') as f:
            train_features = np.load(f)

        with open(join(BASE_PATH, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                       f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                       'test_pretrained_ViT_features.npy'), 'rb') as f:
            test_features = np.load(f)

        # estimate the number of components
        cov_train_features = np.cov(train_features.T)
        values, vectors = eig(cov_train_features)
        sorted_vals = sorted(values, reverse=True)
        cumsum_vals = np.cumsum(sorted_vals)
        explained_vars = cumsum_vals / cumsum_vals[-1]

        for i, explained_var in enumerate(explained_vars):
            n_components = i
            if explained_var > args['whitening_threshold']:
                break

        print_and_add_to_log("=======================", logging)
        print_and_add_to_log(f"number of components are: {n_components}", logging)
        print_and_add_to_log("=======================", logging)

        pca = PCA(n_components=n_components, svd_solver='full', whiten=True)
        train_features = np.ascontiguousarray(pca.fit_transform(train_features))
        test_features = np.ascontiguousarray(pca.transform(test_features))
        print_and_add_to_log("Whitening ended", logging)

        # build GMM
        dens_model = mixture.GaussianMixture(n_components=1,
                                             max_iter=1000,
                                             verbose=1,
                                             n_init=1)
        dens_model.fit(train_features)
        test_pretrained_samples_likelihood = dens_model.score_samples(test_features)
        print_and_add_to_log("----------------------", logging)

        pretrained_auc = roc_auc_score(anomaly_targets, test_pretrained_samples_likelihood)

        print_and_add_to_log(f"Pretrained AUROC score is: {pretrained_auc}", logging)
        print_and_add_to_log("----------------------", logging)
        results['pretrained_AUROC_scores'].append(pretrained_auc)

        # get finetuned prediction head scores
        FINETUNED_PREDICTION_FILE_NAME = 'full_test_finetuned_scores.npy'

        if args['use_imagenet']:
            VIT_MODEL_NAME = 'B_16_imagenet1k'
        else:
            VIT_MODEL_NAME = 'B_16'

        # load best instance
        # Build model
        if not os.path.exists(join(base_feature_path, 'features_distances',
                                   FINETUNED_PREDICTION_FILE_NAME)):

            print_and_add_to_log("Load Model", logging)
            model_checkpoint_path = join(model_path, 'best_full_finetuned_model_state_dict.pkl')
            model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
            model.fc = Identity()
            model_state_dict = torch.load(model_checkpoint_path)
            ret = model.load_state_dict(model_state_dict)
            print_and_add_to_log(
                'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys),
                logging)
            print_and_add_to_log(
                'Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys),
                logging)
            print_and_add_to_log("model loadded from checkpoint here:", logging)
            print_and_add_to_log(model_checkpoint_path, logging)
            model = model.to('cuda')
            model.eval()

            test_finetuned_features = get_finetuned_features(model=model,
                                                             loader=test_loader)

            if not os.path.exists(join(base_feature_path, 'features_distances')):
                os.makedirs(join(base_feature_path, 'features_distances'))
            np.save(join(base_feature_path, 'features_distances', FINETUNED_PREDICTION_FILE_NAME),
                    test_finetuned_features)

        else:
            test_finetuned_features = np.load(
                join(base_feature_path, 'features_distances', FINETUNED_PREDICTION_FILE_NAME))

        if test_finetuned_features.shape[0] == 1:
            test_finetuned_features = test_finetuned_features[0]

        if args["use_layer_outputs"] is None:
            args["use_layer_outputs"] = list(range(test_finetuned_features.shape[1]))

        if not os.path.exists(join(base_feature_path,
                                   'features_distances', 'train_finetuned_features.npy')):
            print_and_add_to_log("Load Model", logging)
            model_checkpoint_path = join(model_path, 'best_full_finetuned_model_state_dict.pkl')
            model = AnomalyViT(VIT_MODEL_NAME, pretrained=True)
            model.fc = Identity()
            model_state_dict = torch.load(model_checkpoint_path)
            ret = model.load_state_dict(model_state_dict)
            print_and_add_to_log(
                'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys),
                logging)
            print_and_add_to_log(
                'Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys),
                logging)
            print_and_add_to_log("model loadded from checkpoint here:", logging)
            print_and_add_to_log(model_checkpoint_path, logging)
            model = model.to('cuda')

            train_finetuned_features = get_finetuned_features(model=model,
                                                              loader=train_loader)
            np.save(join(base_feature_path, 'features_distances', 'train_finetuned_features.npy'),
                    train_finetuned_features)
        else:
            train_finetuned_features = np.load(join(base_feature_path, 'features_distances',
                                                    'train_finetuned_features.npy'))

        if train_finetuned_features.shape[0] == 1:
            train_finetuned_features = train_finetuned_features[0]
            print_and_add_to_log("squeeze training features", logging)

        train_finetuned_features = train_finetuned_features[:, args['use_layer_outputs']]
        test_finetuned_features = test_finetuned_features[:, args['use_layer_outputs']]
        gmm_scores = []
        train_gmm_scores = []
        gmm = mixture.GaussianMixture(n_components=1,
                                      max_iter=500,
                                      verbose=1,
                                      n_init=1)
        gmm.fit(train_finetuned_features)
        test_finetuned_samples_likelihood = gmm.score_samples(test_finetuned_features)
        # train_finetuned_samples_likelihood = gmm.score_samples(train_finetuned_features)
        # max_train_finetuned_features = np.max(np.abs(train_finetuned_samples_likelihood), axis=0)

        test_finetuned_auc = roc_auc_score(anomaly_targets, test_finetuned_samples_likelihood)
        print_and_add_to_log(f"All Block outputs prediciton AUROC score is: {test_finetuned_auc}",
                             logging)
        results['all_layers_finetuned_AUROC_scores'].append(test_finetuned_auc)

        print_and_add_to_log("----------------------", logging)

        finetuned_and_pretrained_samples_likelihood = [
            test_finetuned_samples_likelihood[i] + test_pretrained_samples_likelihood[i] for i in
            range(len(test_pretrained_samples_likelihood))]

        finetuned_and_pretrained_auc = roc_auc_score(anomaly_targets,
                                                     finetuned_and_pretrained_samples_likelihood)
        print_and_add_to_log(
            f"The bgm and output prediction prediciton AUROC is: {finetuned_and_pretrained_auc}",
            logging)

        results['pretrained_and_finetuned_AUROC_scores'].append(finetuned_and_pretrained_auc)

    results_pd = pd.DataFrame.from_dict(results)
    results_dict_path = join(BASE_PATH,
                             f'summarize_results/{args["dataset"]}/{args["dataset"]}_results.csv')
    if not os.path.exists(join(BASE_PATH, f'summarize_results/{args["dataset"]}')):
        os.makedirs(join(BASE_PATH, f'summarize_results/{args["dataset"]}'))
    results_pd.to_csv(results_dict_path)
