import torch
import argparse
from network.base_models import *
from torchvision import models
from network.ResNet import *
from pi_noise_generator import *
from data_loader import *
import utils
import sys
import yaml
import os
import timm
import time


def get_args():
    # Training settingsc
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', 
                        help="configuration file")
    parser.add_argument('--dataset', type=str, default='fashion-mnist',
                        help='dataset name.')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='Whether use pretrained weight.')
    parser.add_argument('--network', type=str, default='dnn3',
                        help='network type.')
    parser.add_argument('--generator', type=str, default='dnn3',
                        help='noise generator type.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--cuda_number', type=str, default='cuda:0',
                        help='Cuda number')
    parser.add_argument('--coeff_category', type=float, default=0.01,
                        help='the coefficient of category in genrator')
    parser.add_argument('--noise_size', type=int, default=1,
                        help='Number of Noise for each sample')
    parser.add_argument('--exp_name', type=str, default="",
                        help='The name of experiment')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lamda', type=float, default=1e-4,
                        help='loss parameter.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for train')
    parser.add_argument('--seed', type=float, default=1,
                        help='random seed')
    parser.add_argument('--optim', type=str, default='SGD',
                        help='network type.')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

def train(base_model, noise_generator, data_loader, test_data_loader, args):
    base_model.train()
    noise_generator.train()
    if cmd_args.optim == "adam":
        optimizer = torch.optim.Adam([{'params': base_model.parameters()}, {'params': noise_generator.parameters()}], lr=args.lr, weight_decay=args.weight_decay)  
    else:
        optimizer = torch.optim.SGD([{'params': base_model.parameters()}, {'params': noise_generator.parameters()}], 
                                lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    cross_entropy = torch.nn.CrossEntropyLoss()
    if not cmd_args.optim == "adam":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cmd_args.epochs)
    for epoch in range(args.epochs):
        epoch_start_time = time.time() 
        for i, data in enumerate(data_loader, 0):
            batch_data, batch_labels = data 
            if batch_data.device != device:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            #data + labels to make noise
            batch_data_labels = batch_labels / (n_class - 1) * cmd_args.coeff_category
            batch_data_labels = batch_data_labels.expand(batch_data.size(1), batch_data.size(2), batch_data.size(3), -1).permute(3, 0, 1, 2)
            batch_data_labels = batch_data_labels + batch_data

            optimizer.zero_grad()
            mu, variance = noise_generator(batch_data_labels)
            # noises is (batch_size, num, dim)
            noises = noise_generator.sampling(mu, variance, args.noise_size)
            expanded_data = batch_data.expand(args.noise_size, -1, -1, -1, -1).permute(1, 0, 2, 3, 4)
            # After reshaping, labels should change from [1, 2, 3, ...] to [1, 1, ..., 2, 2, ..., 3, 3, 3 ...]
            expanded_labels = batch_labels.expand(args.noise_size, -1).t().reshape(batch_labels.size(0) * args.noise_size)
            # input_data is (batch_size * num, dim)
            input_data = expanded_data + noises
            input_data = input_data.reshape(input_data.size(0)*input_data.size(1), input_data.size(2), input_data.size(3), input_data.size(4))
            output = base_model(input_data)
            loss = cross_entropy(output, expanded_labels)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 5 == 0:
            acc_train, loss_train = test(base_model, noise_generator, data_loader)
            logger.info('train Epoch:[{}/{}]\t loss={:.5f}\t acc={:.5f}'.format(epoch , args.epochs, loss_train, acc_train))
            acc_test = test2(base_model, noise_generator, test_data_loader)
            logger.info('test  Epoch:[{}/{}]\t acc={:.5f}'.format(epoch , args.epochs, acc_test))
            base_model.train()
            noise_generator.train()
        if not cmd_args.optim == "adam":
            scheduler.step()
        epoch_end_time = time.time() 
        logger.info(f"Time: {epoch_end_time - epoch_start_time:.2f} seconds")

#test accuracy and loss for training dataset
def test(base_model, noise_generator, data_loader):
    base_model.eval()
    if noise_generator is not None:
        noise_generator.eval()
    cross_entropy = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            batch_data, batch_labels = data
            noise = 0
            if batch_data.device != device:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            if noise_generator is not None:
                #data + labels to make noise
                batch_data_labels = torch.tensor(batch_labels / (n_class - 1) * cmd_args.coeff_category)
                batch_data_labels = batch_data_labels.expand(batch_data.size(1), batch_data.size(2), batch_data.size(3), -1).permute(3, 0, 1, 2)
                batch_data_labels = batch_data_labels + batch_data
                mu, variance = noise_generator(batch_data_labels)
                noise = noise_generator.sampling(mu, variance, 1)
                noise = noise.reshape(noise.size(0), noise.size(2), noise.size(3), noise.size(4))
            output = base_model(batch_data+noise)
            loss += cross_entropy(output, batch_labels.long()).item()
            soft_labels = output.softmax(dim=1)
            _, pred = soft_labels.max(dim=1)
            correct += (pred == batch_labels).double().sum()
            total += len(pred)
            
    return correct/total, loss/(i+1)

#test accuracy for test dataset
def test2(base_model, noise_generator, data_loader):
    base_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            batch_data, batch_labels = data
            noise = 0
            if batch_data.device != device:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            if noise_generator is not None:
                noise_generator.eval()
                soft_labels = torch.empty(batch_data.size(0), n_class).to(batch_data)
                for pred in range(n_class):
                    batch_data_labels = pred / (n_class - 1) * cmd_args.coeff_category
                    batch_data_labels =  batch_data + batch_data_labels
                    mu, variance = noise_generator(batch_data_labels)
                    noise = noise_generator.sampling(mu, variance, 1)
                    noise = noise.reshape(noise.size(0), noise.size(2), noise.size(3), noise.size(4))
                    output = base_model(batch_data+noise)
                    soft_label = output.softmax(dim=1)
                    soft_labels[:, pred] = soft_label[:, pred]
            else:
                output = base_model(batch_data)
                soft_labels = output.softmax(dim=1)

            _, pred = soft_labels.max(dim=1)
            correct += (pred == batch_labels).double().sum()
            total += len(pred)
    return correct/total

if __name__ == "__main__":
    cmd_args = get_args()
    # process argparse & yaml
    if  cmd_args.config:
        opt = vars(cmd_args)
        args = yaml.load(open(cmd_args.config), Loader=yaml.FullLoader)
        opt.update(args)
        cmd_args = opt
    # set seed
    if cmd_args.seed is not None:
        utils.set_seed(cmd_args.seed)
    device = torch.device(cmd_args.cuda_number) if cmd_args.cuda else torch.device('cpu')

    if cmd_args.dataset == 'tiny-imagenet':
        training_dataset = TinyImageNet(train=True)
        test_dataset = TinyImageNet(train=False)
        input_dim = 64
        n_class = 200
        n_channel = 3
    elif cmd_args.dataset == 'imagenet-1k':
        training_dataset, test_dataset, input_dim, n_class, n_channel = load_imagenet()
    else:
        training_X, training_Y, test_X, test_Y, transform_train, transform_test = load_data(cmd_args.dataset, device)
        training_dataset = VPNDataset(training_X, training_Y, transform_train)
        test_dataset = VPNDataset(test_X, test_Y, transform_test)
        
        input_dim = training_dataset.get_feature_dim()
        n_class = training_dataset.get_n_class()
        n_channel = training_dataset.get_n_channel()
    
    training_data_loader = torch.utils.data.DataLoader(training_dataset, shuffle=True,
                                                        batch_size=cmd_args.batch_size, num_workers=16, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                                                    batch_size=cmd_args.batch_size, num_workers=16, pin_memory=True)
    #make experiment directory
    exp_name, exp_dir, log_file, config_file, param_dir, = utils.exp_init(
        cmd_args.exp_name, cmd_args.dataset, cmd_args.network, cmd_args.generator
    )

    # save config
    with open(config_file, "w") as f:
        yaml.dump(cmd_args, f)
    #get logger
    logger = utils.get_logger(log_file)

    if cmd_args.network == 'softmax':
        model = SoftmaxRegression(n_channel, input_dim, n_class)
    elif cmd_args.network == 'dnn3':
        model = SimpleNN3(n_channel, input_dim, n_class)
    elif cmd_args.network == 'resnet18':
        if cmd_args.pretrain == False:
            model = ResNet18(n_class, n_channel)
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            if not cmd_args.dataset == 'imagenet-1k':
                numFit = model.fc.in_features
                model.fc = nn.Linear(numFit, n_class)
    elif cmd_args.network == 'resnet34':
        if cmd_args.pretrain == False:
            model = ResNet34(n_class, n_channel)
        else:
            model = models.resnet34(weights=True)
            if not cmd_args.dataset == 'imagenet-1k':
                numFit = model.fc.in_features
                model.fc = nn.Linear(numFit, n_class)
    elif cmd_args.network == 'resnet50':
        if cmd_args.pretrain == False:
            model = ResNet50(n_class, n_channel)
        else:
            model = models.resnet50(weights=True)
            if not cmd_args.dataset == 'imagenet-1k':
                numFit = model.fc.in_features
                model.fc = nn.Linear(numFit, n_class)
    else:
        print("no network named {}".format(cmd_args.network))
        sys.exit()
    model.to(device)

    #noise generator network
    if cmd_args.generator == 'dnn3':
        gaussian_noise_generator = GaussianNoiseGeneratorDNN3(n_channel, input_dim)
    elif cmd_args.generator == 'resnet18':
        gaussian_noise_generator = GaussianNoiseGeneratorResnet18(n_channel, input_dim)
    gaussian_noise_generator.to(device)

    train(model, gaussian_noise_generator, training_data_loader, test_data_loader, cmd_args)
    torch.save(model, param_dir + '/classifier.pth')
    torch.save(gaussian_noise_generator, param_dir + '/noiseGenerator.pth')