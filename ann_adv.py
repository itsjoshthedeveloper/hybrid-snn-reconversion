#############################
#   @author: Nitin Rathi    #
#############################
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
# from torchviz import make_dot
from matplotlib import pyplot as plt
import pdb
import sys
import datetime
import os
from self_models import *

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
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def train(epoch, loader, epsilon, adv_data):

    global learning_rate
    adv_batch = False
    
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']
    
    #total_correct   = 0
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        
        start_time = datetime.datetime.now()

        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()

        if adv_batch == False or adv_data == False:
            perturbed_data = data
        else:
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # print('\nData: {}\nTarget: {}\n'.format(data, target))
            # Forward pass the data through the model
            output, _ = model(data)
            # print('Output: {}'.format(output))
            # print('Output.max: {}'.format(output.max(1, keepdim=True)))
            # print('Output.max[1]: {}'.format(output.max(1, keepdim=True)[1].reshape(1, -1).squeeze()))
            init_pred = output.max(1, keepdim=True)[1].reshape(1, -1).squeeze() # get the index of the max log-probability

            # print('\nInitial_pred.item(): {}\nTarget.item(): {}\n'.format(init_pred.item(), target.item()))
            # # If the initial prediction is wrong, dont bother attacking, just move on
            # if init_pred.item() != target.item():
            #     continue

            # Calculate the loss
            loss = F.nll_loss(output, target)
            # print('Loss: {}'.format(loss))

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            # print('Perturbed data: {}'.format(perturbed_data))

        optimizer.zero_grad()

        # Re-classify the perturbed image
        output, activations_adv = model(perturbed_data)

        loss = F.cross_entropy(output,target)
        #make_dot(loss).view()
        #exit(0)
        loss.backward()
        optimizer.step()
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        #total_correct += correct.item()

        losses.update(loss.item(), data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))

        # print('Adversarial batch? {} \tBatch #{}'.format(adv_batch, batch_count))
        if adv_data:
            adv_batch = not adv_batch
        
    f.write('\n Epoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
            epoch,
            learning_rate,
            losses.avg,
            top1.avg
            )
        )

def test(loader, epsilon, adv_data):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    model.eval()
    total_loss = 0
    correct = 0
    global max_accuracy, start_time
    
    for batch_idx, (data, target) in enumerate(loader):
                    
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()

        if adv_data == False:
            perturbed_data = data
        else:
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # print('\nData: {}\nTarget: {}\n'.format(data, target))
            # Forward pass the data through the model
            output, _ = model(data)
            # print('Output: {}'.format(output))
            # print('Output.max: {}'.format(output.max(1, keepdim=True)))
            # print('Output.max[1]: {}'.format(output.max(1, keepdim=True)[1].reshape(1, -1).squeeze()))
            init_pred = output.max(1, keepdim=True)[1].reshape(1, -1).squeeze() # get the index of the max log-probability

            # print('\nInitial_pred.item(): {}\nTarget.item(): {}\n'.format(init_pred.item(), target.item()))
            # # If the initial prediction is wrong, dont bother attacking, just move on
            # if init_pred.item() != target.item():
            #     continue

            # Calculate the loss
            loss = F.nll_loss(output, target)
            # print('Loss: {}'.format(loss))

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad)
            # print('Perturbed data: {}'.format(perturbed_data))

        optimizer.zero_grad()
        
        # Re-classify the perturbed image
        output, activations_adv = model(perturbed_data)

        loss = F.cross_entropy(output,target)
        total_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        losses.update(loss.item(), data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))

    if epoch>30 and top1.avg<0.15:
        f.write('\n Quitting as the training is not progressing')
        exit(0)

    if top1.avg>max_accuracy:
        max_accuracy = top1.avg
        state = {
                'accuracy'      : max_accuracy,
                'epoch'         : epoch,
                'state_dict'    : model.state_dict(),
                'optimizer'     : optimizer.state_dict()
        }
        try:
            os.mkdir('./trained_models/ann_adv/')
        except OSError:
            pass
        
        filename = './trained_models/ann_adv/'+identifier+'.pth'
        torch.save(state,filename)
        
    f.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f}, time: {}'.  format(
        losses.avg, 
        top1.avg,
        max_accuracy,
        datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
        )
    )
    # f.write('\n Time: {}'.format(
    #     datetime.timedelta(seconds=(datetime.datetime.now() - current_time).seconds)
    #     )
    # )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ANN to be later converted to SNN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19','RESNET12','RESNET20','RESNET34'])
    parser.add_argument('-lr','--learning_rate',    default=1e-2,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained model to initialize ANN')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--epochs',                 default=300,                type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--optimizer',              default='SGD',              type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')
    parser.add_argument('--momentum',               default=0.9,                type=float,     help='momentum parameter for the SGD optimizer')
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.2,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
    parser.add_argument('-eps','--epsilon',         default=8/255,              type=float,     help='epsilon to train at')
    parser.add_argument('--adv_data',               default=True,              type=bool,       help='enable training with half clean and half adversarial data')
    
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
    learning_rate   = args.learning_rate
    pretrained_ann  = args.pretrained_ann
    epochs          = args.epochs
    lr_reduce       = args.lr_reduce
    optimizer       = args.optimizer
    weight_decay    = args.weight_decay
    momentum        = args.momentum
    amsgrad         = args.amsgrad
    dropout         = args.dropout
    kernel_size     = args.kernel_size
    epsilon         = args.epsilon
    adv_data        = args.adv_data

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))
    
    
    log_file = './logs/ann_adv/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 
    
    #identifier = 'ann_'+architecture.lower()+'_'+dataset.lower()+'_'+str(datetime.datetime.now())
    # get current time
    now = datetime.datetime.now() # current date and time
    date_time = now.strftime('%m-%d-%Y_%H-%M-%S')

    identifier = date_time+'_ann_'+architecture.lower()+'_'+dataset.lower()
    log_file+=identifier+'.log'
    
    if args.log:
        f= open(log_file, 'w', buffering=1)
    else:
        f=sys.stdout
    
    
    f.write('\n Run on time: {}'.format(now))
            
    f.write('\n\n Arguments:')
    for arg in vars(args):
        if arg == 'lr_interval':
            f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
        
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
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
        ])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    
    if dataset == 'CIFAR100':
        train_dataset   = datasets.CIFAR100(root='~/Datasets/cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'CIFAR10': 
        train_dataset   = datasets.CIFAR10(root='~/Datasets/cifar_data', train=True, download=True,transform =transform_train)
        test_dataset    = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform=transform_test)
    
    elif dataset == 'MNIST':
        train_dataset   = datasets.MNIST(root='~/Datasets/mnist/', train=True, download=True, transform=transforms.ToTensor()
            )
        test_dataset    = datasets.MNIST(root='~/Datasets/mnist/', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader    = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader     = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
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
    
    #CIFAR100 sometimes has problem to start training
    #One solution is to train for CIFAR10 with same architecture
    #Load the CIFAR10 trained model except the final layer weights
    model = nn.DataParallel(model)
    if args.pretrained_ann:
        state=torch.load(args.pretrained_ann, map_location='cpu')
        cur_dict = model.state_dict()
        for key in state['state_dict'].keys():
            if key in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[key].shape):
                    cur_dict[key] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
                else:
                    f.write('\n Error: Size mismatch, size of loaded model {}, size of current model {}'.format(state['state_dict'][key].shape, model.state_dict()[key].shape))
            else:
                f.write('\n Error: Loaded weight {} not present in current model'.format(key))
        
        model.load_state_dict(cur_dict)
    
    f.write('\n {}'.format(model)) 
    
    if torch.cuda.is_available() and args.gpu:
        model.cuda()
    
    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay)
    
    f.write('\n {}'.format(optimizer))
    max_accuracy = 0
    
    for epoch in range(1, epochs):
        start_time = datetime.datetime.now()
        train(epoch, train_loader, epsilon, adv_data)
        test(test_loader, epsilon, adv_data)

    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))
    f.write('\n\n Total script time: {}'.format(datetime.timedelta(days=(datetime.datetime.now() - now).days, seconds=(datetime.datetime.now() - now).seconds)))
    f.close()