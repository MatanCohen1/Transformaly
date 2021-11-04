# Imports
from sklearn.decomposition import PCA
from torch.utils.data import Dataset
from utils import *

if __name__ == '__main__':

    args = {}

    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--dataset', default='cifar10')
    # # parser.add_argument('--diag_path', default='./data/fisher_diagonal.pth', help='fim diagonal path')
    # # parser.add_argument('--ewc', action='store_true', help='Train with EWC')
    # # parser.add_argument('--epochs', default=15, type=int, metavar='epochs', help='number of epochs')
    # # parser.add_argument('--lr', type=float, default=1e-2, help='The initial learning rate.')
    # # parser.add_argument('--batch_size', default=20, type=int)
    # # parser.add_argument('--treat-as-abnormal', default=False, action='store_true', help='rest vs 1 settings')
    # parser_args = parser.parse_args()
    # args = vars(parser_args)

    args['dataset'] = 'cifar10'

    args['train_model'] = False
    args['eval_model'] = True


    args['epochs'] = 5
    args['early_stopping_n_epochs'] = 30
    args['eval_every'] = 2  # Will evaluate the model ever <eval_every> epochs
    args['sample_num'] = 1

    args['use_layer_outputs'] = list(range(2, 12))
    args['unimodal'] = True
    args['whitening_threshold'] = 0.9
    args['use_imagenet'] = True
    args['plot_every_layer_summarization'] = True
    args['data_path'] = '/home/access/thesis/anomaly_detection/data'

    args['batch_size'] = 5
    args['lr'] = 0.0001

    args['model_epoch'] = -1

    base_path = 'experiments'


    if args['dataset'] == 'cifar10':
        number_of_classes = range(10)
    elif args['dataset'] == 'fmnist':
        number_of_classes = range(10)
    elif args['dataset'] == 'cifar100':
        number_of_classes = range(20)
    elif args['dataset'] == 'cats_vs_dogs':
        number_of_classes = range(2)
    elif args['dataset'] == 'dior':
        number_of_classes = range(19)
    else:
        raise ValueError(f"Does not support the {args['dataset']} dataset")

    logging.basicConfig(filename=join(base_path,
                                      f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}',
                                      f'{args["dataset"]}_Transformaly_outputs.log'), level=logging.DEBUG)

    print_and_add_to_log("========================================================", logging)
    print_and_add_to_log("Args are:", logging)
    print_and_add_to_log(args, logging)
    print_and_add_to_log("========================================================", logging)
    results = {'class': [],
               'pretrained_AUROC_scores': [],
               'all_layers_finetuned_AUROC_scores': [],
               'pretrained_and_finetuned_AUROC_scores': []}

    #create the relevant directories
    if not os.path.exists(join(base_path, f'{"unimodal" if args["unimodal"] else "multimodal" }/{args["dataset"]}')):
        os.makedirs(join(base_path, f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}'))


    for _class in [0]:
        print_and_add_to_log("===================================", logging)
        print_and_add_to_log(f"Class is : {_class}", logging)
        print_and_add_to_log("===================================", logging)
        args['_class'] = _class
        base_feature_path = join(base_path,
                                 f'{"unimodal" if args["unimodal"] else "multimodal"}/{args["dataset"]}/class_{_class}')
        model_path = join(base_feature_path, 'model')

        args['base_feature_path'] = base_feature_path
        args['model_path'] = model_path

        # create the relevant directories
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if args['unimodal']:
            anomaly_classes = [i for i in number_of_classes if i != args['_class']]
        else:
            anomaly_classes = [args['_class']]

        if args['train_model']:
            print_and_add_to_log(
                "===========================================================================================================================", logging)
            print_and_add_to_log(
                "Start Training", logging)
            print_and_add_to_log(
                "===========================================================================================================================", logging)

            trainset, testset = get_datasets_for_ViT(dataset=args['dataset'],
                                                     data_path = args['data_path'],
                                                     one_vs_rest=args['unimodal'],
                                                     _class=args['_class'],
                                                     normal_test_sample_only=True,
                                                     use_imagenet=args['use_imagenet']
                                                     )

            _, ood_test_set = get_datasets_for_ViT(dataset=args['dataset'],
                                                   data_path = args['data_path'],
                                                   one_vs_rest=not args['unimodal'],
                                                   _class=args['_class'],
                                                   normal_test_sample_only=True,
                                                   use_imagenet=args['use_imagenet']
                                                   )

            print_and_add_to_log("---------------", logging)
            print_and_add_to_log(f'Class size: {args["_class"]}', logging)
            print_and_add_to_log(f'Trainset size: {len(trainset)}', logging)
            print_and_add_to_log(f'Testset size: {len(testset)}', logging)
            print_and_add_to_log(f'OOD testset size: {len(ood_test_set)}', logging)
            print_and_add_to_log("---------------", logging)

            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True)
            val_loader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False)
            ood_val_loader = torch.utils.data.DataLoader(ood_test_set, batch_size=args['batch_size'], shuffle=False)

            dataloaders = {'training': train_loader,
                           'val': val_loader,
                           'test': ood_val_loader
                           }

            # Build model
            if args['use_imagenet']:
                vit_model_name = 'B_16_imagenet1k'
            else:
                vit_model_name = 'B_16'

            # Build model
            model = AnomalyViT(vit_model_name, pretrained=True)
            model.fc = Identity()
            # Build model for best instance
            best_model = AnomalyViT(vit_model_name, pretrained=True)
            best_model.fc = Identity()

            model.to('cuda')
            best_model.to('cuda')

            model_checkpoint_path = join(model_path, f'last_full_finetuned_model_state_dict.pkl')
            if os.path.exists(model_checkpoint_path):
                model_state_dict = torch.load(model_checkpoint_path)
                model.load_state_dict(model_state_dict)
                print_and_add_to_log("model loadded from checkpoint here:", logging)
                print_and_add_to_log(model_checkpoint_path, logging)

            #freeze the model
            freeze_finetuned_model(model)
            model, best_model, cur_acc_loss = train(model=model,
                                                    best_model=best_model,
                                                    args=args,
                                                    dataloaders=dataloaders,
                                                    output_path=model_path,
                                                    device='cuda',
                                                    seed=42,
                                                    model_checkpoint_path=model_checkpoint_path,
                                                    anomaly_classes=anomaly_classes
                                                    )

            training_losses = cur_acc_loss['training_losses']
            val_losses = cur_acc_loss['val_losses']
            try:
                plot_graphs(training_losses, val_losses, training_losses, val_losses)

            except Exception as e:
                print_and_add_to_log('raise error:', logging)
                print_and_add_to_log(e, logging)

            # save models
            torch.save(best_model.state_dict(), join(model_path, f'best_full_finetuned_model_state_dict.pkl'))
            torch.save(model.state_dict(), join(model_path, f'last_full_finetuned_model_state_dict.pkl'))

            # save losses
            with open(join(model_path, f'full_finetuned_training_losses.pkl'), 'wb') as f:
                pickle.dump(training_losses, f)
            with open(join(model_path, f'full_finetuned_val_losses.pkl'), 'wb') as f:
                pickle.dump(val_losses, f)


            if args['use_imagenet']:
                model_name = 'B_16_imagenet1k'
            else:
                model_name = 'B_16'


            model = ViT(model_name, pretrained=True)
            model.fc = Identity()
            model.eval()

            extract_fetures(base_path=base_path,
                            data_path=args['data_path'],
                            datasets=[args['dataset']],
                            model=model,
                            logging = logging,
                            calculate_features=True,
                            unimodal_vals=[args['unimodal']],
                            output_train_features=True,
                            output_test_features=True,
                            use_imagenet=args['use_imagenet'])

        if args['eval_model']:
            results['class'].append(args['_class'])

            trainset, testset = get_datasets_for_ViT(dataset=args['dataset'],
                                                     data_path=args['data_path'],
                                                     one_vs_rest=args['unimodal'],
                                                     _class=args['_class'],
                                                     normal_test_sample_only=False,
                                                     use_imagenet=args['use_imagenet'],
                                                     )

            test_loader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=False)

            anomaly_targets = [0 if i in anomaly_classes else 1 for i in testset.targets]

            print_and_add_to_log("=====================================================")


            # get ViT features
            with open(join(base_path,f'{"unimodal" if args["unimodal"] else "multimodal"}',
                           f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                           f'train_pretrained_ViT_features.npy'), 'rb') as f:
                train_features = np.load(f)

            with open(join(base_path, f'{"unimodal" if args["unimodal"] else "multimodal"}',
                           f'{args["dataset"]}/class_{args["_class"]}/extracted_features',
                           f'test_pretrained_ViT_features.npy'), 'rb') as f:
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
            print_and_add_to_log(f"Whitening ended", logging)

            # build GMM
            dens_model = mixture.GaussianMixture(n_components=1,
                                                 max_iter=1000,
                                                 verbose=1,
                                                 n_init=1)
            dens_model.fit(train_features)
            test_pretrained_samples_likelihood = dens_model.score_samples(test_features)
            print_and_add_to_log("----------------------")

            pretrained_auc = roc_auc_score(anomaly_targets, test_pretrained_samples_likelihood)
            
            print_and_add_to_log(f"Pretrained AUROC score is: {pretrained_auc}", logging)
            print_and_add_to_log("----------------------", logging)
            results['pretrained_AUROC_scores'].append(pretrained_auc)

            # get finetuned prediction head scores
            if args['model_epoch'] > 0:
                finetuned_predcition_file_name = f'{args["model_epoch"]}_epoch_full_test_finetuned_scores.npy'
            else:
                finetuned_predcition_file_name = f'full_test_finetuned_scores.npy'

            if args['use_imagenet']:
                vit_model_name = 'B_16_imagenet1k'
            else:
                vit_model_name = 'B_16'


            # load best instance
            # Build model
            if not os.path.exists(join(base_feature_path, 'features_distances', finetuned_predcition_file_name)):

                if args['model_epoch'] > 0:
                    model_checkpoint_path = join(model_path,
                                                 f'{args["model_epoch"]}_full_finetuned_model_state_dict.pkl')
                else:
                    model_checkpoint_path = join(model_path, f'best_full_finetuned_model_state_dict.pkl')

                print_and_add_to_log("Load Model")
                model = AnomalyViT(vit_model_name, pretrained=True)
                model.fc = Identity()
                model_state_dict = torch.load(model_checkpoint_path)
                ret = model.load_state_dict(model_state_dict)
                print_and_add_to_log('Missing keys when loading pretrained weights: {}'.format(ret.missing_keys), logging)
                print_and_add_to_log('Unexpected keys when loading pretrained weights: {}'.format(ret.unexpected_keys), logging)
                print_and_add_to_log("model loadded from checkpoint here:", logging)
                print_and_add_to_log(model_checkpoint_path, logging)
                model = model.to('cuda')

                test_finetuned_features = get_finetuned_features(args=args,
                                                                  model=model,
                                                                  loader=test_loader)

                if not os.path.exists(join(base_feature_path, 'features_distances')):
                    os.makedirs(join(base_feature_path, 'features_distances'))
                np.save(join(base_feature_path, 'features_distances', finetuned_predcition_file_name),
                        test_finetuned_features)

            else:
                test_finetuned_features = np.load(
                    join(base_feature_path, 'features_distances', finetuned_predcition_file_name))

            if test_finetuned_features.shape[0] == 1:
                test_finetuned_features = test_finetuned_features[0]


            if args["use_layer_outputs"] is None:
                args["use_layer_outputs"] = list(range(test_finetuned_features.shape[1]))


            if not os.path.exists(join(base_feature_path, 'features_distances', 'train_finetuned_features.npy')):
                train_finetuned_features = get_finetuned_features(args=args,
                                                                  model=model,
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
            print_and_add_to_log(f"All Block outputs prediciton AUROC score is: {test_finetuned_auc}", logging)
            results['all_layers_finetuned_AUROC_scores'].append(test_finetuned_auc)

            print_and_add_to_log("----------------------", logging)

            finetuned_and_pretrained_samples_likelihood = [test_finetuned_samples_likelihood[i] + test_pretrained_samples_likelihood[i] for i in
                                                            range(len(test_pretrained_samples_likelihood))]


            finetuned_and_pretrained_auc = roc_auc_score(anomaly_targets, finetuned_and_pretrained_samples_likelihood)
            print_and_add_to_log(f"The bgm and output prediction prediciton AUROC is: {finetuned_and_pretrained_auc}", logging)

            results['pretrained_and_finetuned_AUROC_scores'].append(finetuned_and_pretrained_auc)


    results_pd = pd.DataFrame.from_dict(results)
    if args['model_epoch'] > 0:
        results_dict_path = join(base_path,
                                 f'summarize_results/{args["dataset"]}/{args["model_epoch"]}_epoch_{args["dataset"]}_results.csv')
    else:
        results_dict_path = join(base_path, f'summarize_results/{args["dataset"]}/{args["dataset"]}_results.csv')
    if not os.path.exists(join(base_path, f'summarize_results/{args["dataset"]}')):
        os.makedirs(join(base_path, f'summarize_results/{args["dataset"]}'))
    results_pd.to_csv(results_dict_path)