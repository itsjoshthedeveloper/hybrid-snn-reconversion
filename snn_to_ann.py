from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
# from torchviz import make_dot
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import pdb
from self_models import *
import sys
import os
import shutil
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def testFGSM( test_loader, model, device, epsilon, identifier ):

    f.write('\n\n' + identifier)

    # Accuracy counter
    correct = 0
    adv_examples = []
    model.eval()

    # Loop over all examples in test set
    for batch_idx, (data, target) in enumerate(test_loader):

        # # Send the data and label to the device
        # data, target = data.to(device), target.to(device)
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    f.write("\n\tEpsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def test(loader, model, identifier):
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0
        global start_time
        
        for batch_idx, (data, target) in enumerate(loader):

            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = F.cross_entropy(output,target)
            total_loss += loss.item()
            pred = output.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()
            losses.update(loss.item(), data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))

        # if epoch>30 and top1.avg<0.15:
        #     f.write('\n Quitting as the training is not progressing')
        #     exit(0)

        # if top1.avg>max_accuracy:
        #     max_accuracy = top1.avg
        #     state = {
        #             'accuracy'      : max_accuracy,
        #             'epoch'         : epoch,
        #             'state_dict'    : model.state_dict(),
        #             'optimizer'     : optimizer.state_dict()
        #     }
        #     try:
        #         os.mkdir('./trained_models/snn_to_ann/')
        #     except OSError:
        #         pass
        state = {
            'test_loss'     : losses.avg,
            'test_acc'      : top1.avg,
            'time'          : datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds),
            'state_dict'    : model.state_dict()
        }
        try:
            os.mkdir('./trained_models/snn_to_ann/')
        except OSError:
            pass
        filename = './trained_models/snn_to_ann/'+identifier+'.pth'
        torch.save(state, filename)
        
        # f.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f}, time: {}'.  format(
        #     losses.avg, 
        #     top1.avg,
        #     max_accuracy,
        #     datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
        #     )
        # )
        # f.write('\n Time: {}'.format(
        #     datetime.timedelta(seconds=(datetime.datetime.now() - current_time).seconds)
        #     )
        # )

        return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconvert SNN to ANN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu',                    default=False,              type=bool,      help='use gpu')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG5',             type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    # parser.add_argument('-lr','--learning_rate',    default=1e-2,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained model to initialize ANN')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for reconversion to model_prime')
    # parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    # parser.add_argument('--epochs',                 default=300,                type=int,       help='number of training epochs')
    # parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    # parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    # parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    # parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')
    # parser.add_argument('--momentum',               default=0.9,                type=float,     help='momentum parameter for the SGD optimizer')
    # parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.3,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
    
    args=parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
    dataset         = args.dataset
    batch_size      = args.batch_size
    architecture    = args.architecture
    # learning_rate   = args.learning_rate
    pretrained_ann  = args.pretrained_ann
    pretrained_snn  = args.pretrained_snn
    # epochs          = args.epochs
    # lr_reduce       = args.lr_reduce
    # optimizer       = args.optimizer
    # weight_decay    = args.weight_decay
    # momentum        = args.momentum
    # amsgrad         = args.amsgrad
    dropout         = args.dropout
    kernel_size     = args.kernel_size

    # values = args.lr_interval.split()
    # lr_interval = []
    # for value in values:
    #     lr_interval.append(int(float(value)*args.epochs))

    if not pretrained_ann:
        ann_file = './trained_models/ann/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'
        if os.path.exists(ann_file):
            val = input('\n Do you want to use the pretrained ANN {}? Y or N: '.format(ann_file))
            if val.lower()=='y' or val.lower()=='yes':
                pretrained_ann = ann_file
            else:
                print('No pretrained ANN found/loaded')
                sys.exit()
    
    if not pretrained_snn:
        snn_file = './trained_models/snn/snn_'+architecture.lower()+'_'+dataset.lower()+'.pth'
        if os.path.exists(snn_file):
            val = input('\n Do you want to use the pretrained SNN {}? Y or N: '.format(snn_file))
            if val.lower()=='y' or val.lower()=='yes':
                pretrained_snn = snn_file
            else:
                print('No pretrained SNN found/loaded')
                sys.exit()
    
    log_file = './logs/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass
    log_file += 'snn_to_ann/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass
    
    #identifier = 'ann_'+architecture.lower()+'_'+dataset.lower()+'_'+str(datetime.datetime.now())
    ann_identifier = 'ann_'+architecture.lower()+'_'+dataset.lower()
    ann_prime_identifier = 'ann_'+architecture.lower()+'_'+dataset.lower()+'_prime'
    log_file += ann_identifier+'.log'
    
    if args.log:
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout
    
    f.write('\n Run on time: {}'.format(datetime.datetime.now()))
            
    f.write('\n\n Arguments:')
    for arg in vars(args):
        # if arg == 'lr_interval':
        #     f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        # else:
        f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
    f.write('\n\n')
        
    # Training settings
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    # Loading Dataset
    if dataset == 'CIFAR100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
        labels      = 100 
    elif dataset == 'CIFAR10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        labels      = 10
    elif dataset == 'MNIST':
        labels = 10
    
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        # transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # normalize
        # ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    
    if dataset == 'CIFAR100':
        # train_dataset   = datasets.CIFAR100(root='~/Datasets/cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'CIFAR10': 
        # train_dataset   = datasets.CIFAR10(root='~/Datasets/cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'MNIST':
        # train_dataset   = datasets.MNIST(root='~/Datasets/mnist/', train=True, download=True, transform=transforms.ToTensor())
        test_dataset    = datasets.MNIST(root='~/Datasets/mnist/', train=False, download=True, transform=transforms.ToTensor())
    
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    test_loader_fgsm = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    
    if architecture[0:3].lower() == 'vgg':
        model = VGG(vgg_name=architecture, labels=labels, dataset=dataset, kernel_size=kernel_size, dropout=dropout)
    elif architecture[0:3].lower() == 'res':
        if architecture.lower() == 'resnet12':
            model = ResNet12(labels=labels, dropout=dropout)
        elif architecture.lower() == 'resnet20':
            model = ResNet20(labels=labels, dropout=dropout)
        elif architecture.lower() == 'resnet34':
            model = ResNet34(labels=labels, dropout=dropout) 
    #f.write('\n{}'.format(model))
    model_prime = model
    
    #CIFAR100 sometimes has problem to start training
    #One solution is to train for CIFAR10 with same architecture
    #Load the CIFAR10 trained model except the final layer weights
    model = nn.DataParallel(model)
    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')
        cur_dict = model.state_dict()
        for key in state['state_dict'].keys():
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Success [ANN]: Loaded {} from {}'.format(key, pretrained_ann))
                else:
                    f.write('\n Error [ANN]: Size mismatch at {}, size of loaded model {}, size of current model {}'.format(key, state['state_dict'][key].shape, model.state_dict()[key].shape))
            else:
                f.write('\n Error [ANN]: Loaded weight {} not present in current model'.format(key))
        
        model.load_state_dict(cur_dict)
    
    f.write('\n ANN: {}'.format(model)) 

    model_prime = nn.DataParallel(model_prime)
    if pretrained_snn:
        state = torch.load(pretrained_snn, map_location='cpu')
        cur_dict = model_prime.state_dict()
        for key in state['state_dict'].keys():
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Success [ANN\']: Loaded {} from {}'.format(key, pretrained_snn))
                else:
                    f.write('\n Error [ANN\']: Size mismatch at {}, size of loaded model {}, size of current model {}'.format(key, state['state_dict'][key].shape, model_prime.state_dict()[key].shape))
            else:
                f.write('\n Error [ANN\']: Loaded weight {} not present in current model'.format(key))
        
        model_prime.load_state_dict(cur_dict)
    
    f.write('\n {}'.format(model_prime) + '\n')
    
    if torch.cuda.is_available() and args.gpu:
        model.cuda()
        model_prime.cuda()
    
    # if optimizer == 'SGD':
    #     optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # elif optimizer == 'Adam':
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay)
    # f.write('\n {}'.format(optimizer))
    
    # for epoch in range(1, epochs):
    #     start_time = datetime.datetime.now()
    #     test(epoch)

    start_time = datetime.datetime.now()
    ann_state = test(test_loader, model, ann_identifier)
    start_time = datetime.datetime.now()
    ann_prime_state = test(test_loader, model_prime, ann_prime_identifier)

    epsilons = [0, .05, .1, .15, .2, .25, .3]
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    ann_accuracies_fgsm = []
    ann_examples_fgsm = []
    ann_prime_accuracies_fgsm = []
    ann_prime_examples_fgsm = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = testFGSM(test_loader_fgsm, model, device, eps, ann_identifier)
        ann_accuracies_fgsm.append(acc)
        ann_examples_fgsm.append(ex)

    for eps in epsilons:
        acc, ex = testFGSM(test_loader_fgsm, model_prime, device, eps, ann_prime_identifier)
        ann_prime_accuracies_fgsm.append(acc)
        ann_prime_examples_fgsm.append(ex)

    f.write('\n\n ' + ann_identifier)
    f.write('\n\ttest_loss: {:.4f}, test_acc: {:.4f}, time: {}'.format(
            ann_state['test_loss'], 
            ann_state['test_acc'],
            ann_state['time']
            )
        )
    f.write('\n\tepsilons: {}'.format(epsilons))
    f.write('\n\tfgsm accuracies: {}'.format(ann_accuracies_fgsm))

    f.write('\n\n ' + ann_prime_identifier)
    f.write('\n\ttest_loss: {:.4f}, test_acc: {:.4f}, time: {}'.format(
            ann_prime_state['test_loss'], 
            ann_prime_state['test_acc'],
            ann_prime_state['time']
            )
        )
    f.write('\n\tepsilons: {}'.format(epsilons))
    f.write('\n\tfgsm accuracies: {}'.format(ann_prime_accuracies_fgsm))
    
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, ann_accuracies_fgsm, label='ANN')
    plt.plot(epsilons, ann_prime_accuracies_fgsm, label='ANN\'')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon [{}]".format(ann_identifier))
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('./logs/snn_to_ann/'+ann_identifier+'.png', bbox_inches='tight')

    # f.write('\n ANN accuracy: {:.4f}'.format(ann_accuracy))
    # f.write('\n ANN\' accuracy: {:.4f}'.format(ann_prime_accuracy))
    f.close()