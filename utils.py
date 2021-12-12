"""
Transformaly Utils File
"""
# from PIL import Image
import logging
import math
import sys
import os
import gc
import time
from enum import Enum
from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import faiss
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder
from torchvision.transforms import Compose


class DiorDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,
                 image_path,
                 labels_dict_path,
                 transform=None):
        """
        Args:
            image_path (string): Path to the images.
            labels_dict_path (string): Path to the dict with annotations.
        """
        self.image_path = image_path
        self.labels_dict_path = labels_dict_path
        self.transform = transform

        with open(self.labels_dict_path, 'rb') as handle:
            self.labels_dict = pickle.load(handle)
        self.images = [f for f in listdir(image_path) if isfile(join(image_path, f))]
        self.targets = [self.labels_dict[img]['label_index'] for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(join(self.image_path, self.images[idx]))
        if self.transform:
            img = self.transform(img)

        label = self.targets[idx]
        return img, label


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    dist, _ = index.search(test_set, n_neighbours)
    return np.sum(dist, axis=1)


def get_features(model, data_loader, early_break=-1):
    pretrained_features = []
    for i, (data, _) in enumerate(tqdm(data_loader)):
        if early_break > 0 and early_break < i:
            break

        encoded_outputs = model(data.to('cuda'))
        pretrained_features.append(encoded_outputs.detach().cpu().numpy())

    pretrained_features = np.concatenate(pretrained_features)
    return pretrained_features


def freeze_pretrained_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def extract_fetures(base_path,
                    data_path,
                    datasets,
                    model,
                    logging,
                    calculate_features=False,
                    manual_class_num_range=None,
                    unimodal_vals=None,
                    output_train_features=True,
                    output_test_features=True,
                    use_imagenet=False):
    if unimodal_vals is None:
        unimodal_vals = [True, False]

    BATCH_SIZE = 18
    exp_num = -1
    for dataset in datasets:
        print_and_add_to_log("=======================================", logging)
        print_and_add_to_log(f"Dataset: {dataset}", logging)
        print_and_add_to_log(f"Path: {base_path}", logging)
        print_and_add_to_log("=======================================", logging)
        exp_num += 1

        number_of_classes = get_number_of_classes(dataset)

        if manual_class_num_range is not None:
            _classes = range(*manual_class_num_range)

        else:
            _classes = range(number_of_classes)

        for _class in _classes:

            # config
            for unimodal in unimodal_vals:

                print_and_add_to_log("=================================================",
                                     logging)
                print_and_add_to_log(f"Experiment number: {exp_num}", logging)
                print_and_add_to_log(f"Dataset: {dataset}", logging)
                print_and_add_to_log(f"Class: {_class}", logging)
                print_and_add_to_log(f"Unimodal setting: {unimodal}", logging)

                assert dataset in ['cifar10', 'cifar100', 'fmnist', 'cats_vs_dogs',
                                   'dior'], f"{dataset} not supported yet!"
                if unimodal:
                    base_feature_path = join(base_path, f'unimodal/{dataset}/class_{str(_class)}')
                else:
                    base_feature_path = join(base_path, f'multimodal/{dataset}/class_{str(_class)}')

                if not os.path.exists((base_feature_path)):
                    os.makedirs(base_feature_path, )
                else:
                    print_and_add_to_log(f"Experiment of class {_class} already exists", logging)

                if unimodal:
                    anomaly_classes = [i for i in range(number_of_classes) if i != _class]
                else:
                    anomaly_classes = [_class]

                if dataset == 'fmnist':
                    if use_imagenet:
                        val_transforms = Compose(
                            [
                                transforms.Resize((384, 384)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )
                    else:
                        val_transforms = Compose(
                            [
                                transforms.Resize((224, 224)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )
                else:
                    if use_imagenet:
                        val_transforms = Compose(
                            [
                                transforms.Resize((384, 384)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )
                    else:
                        val_transforms = Compose(
                            [
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ]
                        )

                model.eval()
                freeze_pretrained_model(model)
                model.to('cuda')

                # get dataset
                trainset_origin, testset = get_datasets(dataset, data_path, val_transforms)
                indices = [i for i, val in enumerate(trainset_origin.targets)
                           if val not in anomaly_classes]
                trainset = torch.utils.data.Subset(trainset_origin, indices)

                print_and_add_to_log(f"Train dataset len: {len(trainset)}", logging)
                print_and_add_to_log(f"Test dataset len: {len(testset)}", logging)

                # Create datasetLoaders from trainset and testset
                trainsetLoader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
                testsetLoader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

                anomaly_targets = [1 if i in anomaly_classes else 0 for i in testset.targets]

                extracted_features_path = join(base_feature_path, 'extracted_features')
                if not os.path.exists(extracted_features_path):
                    os.makedirs(extracted_features_path)

                print_and_add_to_log("Extracted features", logging)
                if not os.path.exists(extracted_features_path):
                    os.mkdir(extracted_features_path)

                if calculate_features or not os.path.exists(
                        join(extracted_features_path, 'train_pretrained_ViT_features.npy')):
                    if output_train_features:
                        train_features = get_features(model=model, data_loader=trainsetLoader)
                        with open(join(extracted_features_path,
                                       'train_pretrained_ViT_features.npy'), 'wb') as f:
                            np.save(f, train_features)

                    if output_test_features:
                        test_features = get_features(model=model, data_loader=testsetLoader)
                        with open(join(extracted_features_path,
                                       'test_pretrained_ViT_features.npy'), 'wb') as f:
                            np.save(f, test_features)

                else:
                    if output_train_features:
                        print_and_add_to_log(f"loading feature from {extracted_features_path}",
                                             logging)
                        with open(join(extracted_features_path,
                                       'train_pretrained_ViT_features.npy'), 'rb') as f:
                            train_features = np.load(f)
                    if output_test_features:
                        with open(join(extracted_features_path,
                                       f'test_pretrained_ViT_features.npy'), 'rb') as f:
                            test_features = np.load(f)

                if output_train_features and output_test_features:
                    print_and_add_to_log("Calculate KNN score", logging)
                    distances = knn_score(train_features, test_features, n_neighbours=2)
                    auc = roc_auc_score(anomaly_targets, distances)
                    print_and_add_to_log(auc, logging)


def freeze_finetuned_model(model):
    non_freezed_layer = []
    for name, param in model.named_parameters():
        if not (name.startswith('transformer.cloned_block') or name.startswith('cloned_')):
            param.requires_grad = False
        else:
            non_freezed_layer.append(name)
    print("=========================================")
    print("Clone block didn't freezed")
    print(f"layers name: {non_freezed_layer}")
    print("=========================================")
    return


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def forward_one_epoch(loader,
                      optimizer,
                      criterion,
                      net,
                      mode,
                      progress_bar_str,
                      num_of_epochs,
                      device='cuda'
                      ):
    losses = []

    for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):

        if mode == Mode.training:
            optimizer.zero_grad()

        inputs = inputs.to(device)

        origin_block_outputs, cloned_block_outputs = net(inputs)
        loss = criterion(cloned_block_outputs, origin_block_outputs)
        losses.append(loss.item())

        if mode == Mode.training:
            # do a step
            loss.backward()
            optimizer.step()

        if batch_idx % 20 == 0:
            progress_bar(batch_idx, len(loader), progress_bar_str
                         % (num_of_epochs, np.mean(losses), losses[-1]))
        del inputs, origin_block_outputs, cloned_block_outputs, loss
        torch.cuda.empty_cache()

        # if batch_idx > 10:
        #     break
    return losses


def train(model, best_model, args, dataloaders,
          model_checkpoint_path,
          output_path, device='cuda',
          seed=42, anomaly_classes=None):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    best_model = best_model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.MSELoss()

    training_losses, val_losses = [], []

    training_loader = dataloaders['training']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    best_val_loss = np.inf

    # start training
    for epoch in range(1, args['epochs'] + 1):

        # training
        model = model.train()
        progress_bar_str = 'Teain: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'

        losses = forward_one_epoch(loader=training_loader,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   net=model,
                                   mode=Mode.training,
                                   progress_bar_str=progress_bar_str,
                                   num_of_epochs=epoch)

        # save first batch loss for normalization
        train_epoch_loss = np.mean(losses)
        sys.stdout.flush()
        print()
        print(f'Train epoch {epoch}: loss {train_epoch_loss}', flush=True)
        training_losses.append(train_epoch_loss)

        torch.cuda.empty_cache()
        torch.save(model.state_dict(), model_checkpoint_path)

        if epoch == 1 or epoch == 5:
            init_model_checkpoint_path = join(output_path,
                                              f'{epoch}_full_recon_model_state_dict.pkl')
            torch.save(model.state_dict(), init_model_checkpoint_path)

        del losses
        gc.collect()

        if (epoch - 1) % args['eval_every'] == 0:
            # validation
            model.eval()
            progress_bar_str = 'Validation: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'

            losses = forward_one_epoch(loader=val_loader,
                                       optimizer=optimizer,
                                       criterion=criterion,
                                       net=model,
                                       mode=Mode.validation,
                                       progress_bar_str=progress_bar_str,
                                       num_of_epochs=epoch
                                             )

            val_epoch_loss = np.mean(losses)
            sys.stdout.flush()

            print()
            print(f'Validation epoch {epoch // args["eval_every"]}: loss {val_epoch_loss}',
                  flush=True)
            val_losses.append(val_epoch_loss)

            #
            cur_acc_loss = {
                'training_losses': training_losses,
                'val_losses': val_losses
            }

            if best_val_loss - 0.001 > val_epoch_loss:
                best_val_loss = val_epoch_loss
                best_acc_epoch = epoch

                print(f'========== new best model! epoch {best_acc_epoch}, loss {best_val_loss}  ==========')

                best_model.load_state_dict(model.state_dict())
                # best_model = copy.deepcopy(model)
                # no_imporvement_epochs = 0
            # else:
            #     no_imporvement_epochs += 1

            del losses
            gc.collect()

            progress_bar_str = 'Test: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'
            model.eval()
            test_losses = forward_one_epoch(loader=test_loader,
                                            optimizer=None,
                                            criterion=criterion,
                                            net=model,
                                            mode=Mode.test,
                                            progress_bar_str=progress_bar_str,
                                            num_of_epochs=0)

            test_epoch_loss = np.mean(test_losses)
            print("===================== OOD val Results =====================")
            print(f'OOD val Loss : {test_epoch_loss}')
            del test_losses
            gc.collect()
            # if no_imporvement_epochs > args['early_stopping_n_epochs']:
            #     print(f"Stop due to early stopping after {no_imporvement_epochs} epochs without improvment")
            #     print(f"epoch number {epoch}")
            #     break

            if args['plot_every_layer_summarization']:
                _, testset = get_datasets_for_ViT(dataset=args['dataset'],
                                                  data_path = args['data_path'],
                                                  one_vs_rest=args['unimodal'],
                                                  _class=args['_class'],
                                                  normal_test_sample_only=False,
                                                  use_imagenet=args['use_imagenet']
                                                  )

                eval_test_loader = torch.utils.data.DataLoader(testset,
                                                               batch_size=args['batch_size'],
                                                               shuffle=False)
                anomaly_targets = [0 if i in anomaly_classes else 1 for i in testset.targets]

                model = model.eval()
                outputs_recon_scores = get_finetuned_features(model,
                                                              eval_test_loader)
                outputs_recon_scores = outputs_recon_scores[0]

                print("========================================================")
                for j in range(len(args['use_layer_outputs'])):
                    layer_ind = args['use_layer_outputs'][j]
                    print(f"Layer number: {layer_ind}")
                    print(
                        f"Test Max layer outputs score: {np.max(np.abs(outputs_recon_scores[:, layer_ind]))}")
                    rot_auc = roc_auc_score(anomaly_targets,
                                            outputs_recon_scores[:, layer_ind])
                    print(f'layer AUROC score: {rot_auc}')
                    print("--------------------------------------------------------")
            model = model.train()

    progress_bar_str = 'Test: repeat %d -- Mean Loss: %.3f | Last Loss: %.3f'

    model = model.eval()
    test_losses = forward_one_epoch(loader=test_loader,
                                    optimizer=None,
                                    criterion=criterion,
                                    net=model,
                                    mode=Mode.test,
                                    progress_bar_str=progress_bar_str,
                                    num_of_epochs=0)

    best_model = best_model.to('cpu')
    model = model.to('cpu')
    test_epoch_loss = np.mean(test_losses)
    print("===================== OOD val Results =====================")
    print(f'OOD val Loss : {test_epoch_loss}')
    return model, best_model, cur_acc_loss


def get_finetuned_features(model,
                           loader,
                           seed = 42
                           ):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to('cuda')
    criterion = nn.MSELoss(reduce=False)

    # start eval
    model = model.eval()
    progress_bar_str = 'Test: repeat %d -- Mean Loss: %.3f'

    all_outputs_recon_scores = []

    with torch.no_grad():
        outputs_recon_scores = []
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):

            inputs = inputs.to('cuda')

            origin_block_outputs, cloned_block_outputs = model(inputs)
            loss = criterion(cloned_block_outputs, origin_block_outputs)
            loss = torch.mean(loss, [2, 3])
            loss = loss.permute(1, 0)
            outputs_recon_scores.extend(-1 * loss.detach().cpu().data.numpy())

            if batch_idx % 20 == 0:
                progress_bar(batch_idx, len(loader), progress_bar_str
                             % (1, np.mean(outputs_recon_scores)))

            del inputs, origin_block_outputs, cloned_block_outputs, loss
            torch.cuda.empty_cache()
        all_outputs_recon_scores.append(outputs_recon_scores)

    return np.array(all_outputs_recon_scores)


def get_transforms(dataset, use_imagenet):
    # 0.5 normalization
    if dataset == 'fmnist':
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]


    else:
        val_transforms_list = [
            transforms.Resize((384, 384)) if use_imagenet else transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]

    val_transforms = Compose(val_transforms_list)
    return val_transforms


def get_number_of_classes(dataset):
    if dataset == 'cifar10':
        number_of_classes = 10

    elif dataset == 'cifar100':
        number_of_classes = 20

    elif dataset == 'fmnist':
        number_of_classes = 10

    elif dataset == 'cats_vs_dogs':
        number_of_classes = 2

    elif dataset == 'dior':
        number_of_classes = 19

    else:
        raise ValueError(f"{dataset} not supported yet!")
    return number_of_classes


def get_datasets_for_ViT(dataset, data_path, one_vs_rest, _class,
                         normal_test_sample_only=True,
                         use_imagenet=False):
    number_of_classes = get_number_of_classes(dataset)
    if one_vs_rest:
        anomaly_classes = [i for i in range(number_of_classes) if i != _class]
    else:
        anomaly_classes = [_class]

    val_transforms = get_transforms(dataset=dataset,
                                    use_imagenet=use_imagenet)

    # get dataset
    trainset_origin, testset = get_datasets(dataset, data_path, val_transforms)

    train_indices = [i for i, val in enumerate(trainset_origin.targets)
                     if val not in anomaly_classes]
    logging.info(f"len of train dataset {len(train_indices)}")
    trainset = torch.utils.data.Subset(trainset_origin, train_indices)

    if normal_test_sample_only:
        test_indices = [i for i, val in enumerate(testset.targets)
                        if val not in anomaly_classes]
        testset = torch.utils.data.Subset(testset, test_indices)

    logging.info(f"len of test dataset {len(testset)}")
    return trainset, testset


def print_and_add_to_log(msg, logging):
    print(msg)
    logging.info(msg)


def get_datasets(dataset, data_path, val_transforms):
    if dataset == 'cifar100':
        testset = CIFAR100(root=data_path,
                           train=False, download=True,
                           transform=val_transforms)

        trainset = CIFAR100(root=data_path,
                            train=True, download=True,
                            transform=val_transforms)

        trainset.targets = sparse2coarse(trainset.targets)
        testset.targets = sparse2coarse(testset.targets)

    elif dataset == 'cifar10':
        testset = CIFAR10(root=data_path,
                          train=False, download=True,
                          transform=val_transforms)

        trainset = CIFAR10(root=data_path,
                           train=True, download=True,
                           transform=val_transforms)


    elif dataset == 'fmnist':
        trainset = FashionMNIST(root=data_path,
                                train=True, download=True,
                                transform=val_transforms)

        testset = FashionMNIST(root=data_path,
                               train=False, download=True,
                               transform=val_transforms)

    elif dataset == 'cats_vs_dogs':
        trainset = ImageFolder(root=data_path,
                               transform=val_transforms)
        testset = ImageFolder(root=data_path,
                              transform=val_transforms)

    else:
        raise ValueError(f"{dataset} not supported yet!")

    return trainset, testset


class Mode(Enum):
    training = 1
    validation = 2
    test = 3


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except ValueError:
    term_width = 0

term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def plot_graphs(train_accuracies, val_accuracies, train_losses,
                val_losses, path_to_save=''):
    plot_accuracy(train_accuracies, val_accuracies, path_to_save=path_to_save)
    plot_loss(train_losses, val_losses, path_to_save=path_to_save)
    return max(val_accuracies)


def plot_accuracy(train_accuracies, val_accuracies, to_show=True,
                  label='accuracy', path_to_save=''):
    print(f'Best val accuracy was {max(val_accuracies)}, at epoch {np.argmax(val_accuracies)}')
    train_len = len(np.array(train_accuracies))
    val_len = len(np.array(val_accuracies))

    xs_train = list(range(0, train_len))

    if train_len != val_len:
        xs_val = list(range(0, train_len, math.ceil(train_len / val_len)))
    else:
        xs_val = list(range(0, train_len))

    plt.plot(xs_val, np.array(val_accuracies), label='val ' + label)
    plt.plot(xs_train, np.array(train_accuracies), label='train ' + label)
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    if len(path_to_save) > 0:
        plt.savefig(f'{path_to_save}/accuracy_graph.png')

    if to_show:
        plt.show()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def plot_loss(train_losses, val_losses, to_show=True,
              val_label='val loss', train_label='train loss',
              path_to_save=''):
    train_len = len(np.array(train_losses))
    val_len = len(np.array(val_losses))

    xs_train = list(range(0, train_len))
    if train_len != val_len:
        xs_val = list(range(0, train_len, int(train_len / val_len) + 1))
    else:
        xs_val = list(range(0, train_len))

    plt.plot(xs_val, np.array(val_losses), label=val_label)
    plt.plot(xs_train, np.array(train_losses), label=train_label)

    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    if len(path_to_save) > 0:
        plt.savefig(f'{path_to_save}/loss_graph.png')
    if to_show:
        plt.show()
