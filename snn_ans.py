#---------------------------------------------------
# Imports
#---------------------------------------------------
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

def find_threshold(batch_size=512, timesteps=2500, architecture='VGG16'):
    
    loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    model.module.network_update(timesteps=timesteps, leak=1.0)

    pos=0
    thresholds=[]
    
    def find(layer, pos):
        max_act=0
        
        f.write('\n Finding threshold for layer {}'.format(layer))
        for batch_idx, (data, target) in enumerate(loader):
            
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)
                if output>max_act:
                    max_act = output.item()

                #f.write('\nBatch:{} Current:{:.4f} Max:{:.4f}'.format(batch_idx+1,output.item(),max_act))
                if batch_idx==0:
                    thresholds.append(max_act)
                    pos = pos+1
                    f.write(' {}'.format(thresholds))
                    model.module.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])
                    break
        return pos

    if architecture.lower().startswith('vgg'):              
        for l in model.module.features.named_children():
            if isinstance(l[1], nn.Conv2d):
                pos = find(int(l[0]), pos)
        
        for c in model.module.classifier.named_children():
            if isinstance(c[1], nn.Linear):
                if (int(l[0])+int(c[0])+1) == (len(model.module.features) + len(model.module.classifier) -1):
                    pass
                else:
                    pos = find(int(l[0])+int(c[0])+1, pos)

    if architecture.lower().startswith('res'):
        for l in model.module.pre_process.named_children():
            if isinstance(l[1], nn.Conv2d):
                pos = find(int(l[0]), pos)
    f.write('\n ANN thresholds: {}'.format(thresholds))
    return thresholds

def train(epoch):

    global learning_rate

    model.module.network_update(timesteps=timesteps, leak=leak)
    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    if epoch in lr_interval:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / lr_reduce
            learning_rate = param_group['lr']
    
    #f.write('Epoch: {} Learning Rate: {:.2e}'.format(epoch,learning_rate_use))
    
    #total_loss = 0.0
    #total_correct = 0
    model.train()

    #current_time = start_time
    #model.module.network_init(update_interval)

    for batch_idx, (data, target) in enumerate(train_loader):

        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data) 
        #pdb.set_trace()
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()        
        pred = output.max(1,keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        losses.update(loss.item(),data.size(0))
        top1.update(correct.item()/data.size(0), data.size(0))
                
        if (batch_idx+1) % train_acc_batches == 0:
            temp1 = []
            for value in model.module.threshold.values():
                temp1 = temp1+[round(value.item(),2)]
            f.write('\nEpoch: {}, batch: {}, train_loss: {:.4f}, train_acc: {:.4f}, threshold: {}, leak: {}, timesteps: {}'
                    .format(epoch,
                        batch_idx+1,
                        losses.avg,
                        top1.avg,
                        temp1,
                        model.module.leak.item(),
                        model.module.timesteps
                        )
                    )
    f.write('\nEpoch: {}, lr: {:.1e}, train_loss: {:.4f}, train_acc: {:.4f}'
                    .format(epoch,
                        learning_rate,
                        losses.avg,
                        top1.avg,
                        )
                    )

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

def testFGSM( test_loader, model, device, epsilon ):

    # Accuracy counter
    correct = 0
    model.eval()

    # Loop over all examples in test set
    for batch_idx, (data, target) in enumerate(test_loader):

        # # Send the data and label to the device
        # data, target = data.to(device), target.to(device)
        if torch.cuda.is_available() and args.gpu:
            data, target = data.cuda(), target.cuda()

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        if epsilon == 0:
            perturbed_data = data
        else:
            # Forward pass the data through the model
            output, _ = model(data)
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
        output, activations_adv = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
        #     # Special case for saving 0 epsilon examples
        #     if (epsilon == 0) and (len(adv_examples) < 5):
        #         adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        #         adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        # else:
        #     # Save some adv examples for visualization later
        #     if len(adv_examples) < 5:
        #         adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        #         adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    f.write("\n\tEpsilon: {}\tTest Accuracy = {} / {} = {}\tActivation layers: {}".format(epsilon, correct, len(test_loader), final_acc, len(activations_adv)))

    # Return the accuracy and an adversarial example
    return final_acc, activations_adv

def test(epoch):

    losses = AverageMeter('Loss')
    top1   = AverageMeter('Acc@1')

    with torch.no_grad():
        model.eval()
        global max_accuracy
        
        for batch_idx, (data, target) in enumerate(test_loader):
                        
            if torch.cuda.is_available() and args.gpu:
                data, target = data.cuda(), target.cuda()
            
            output  = model(data) 
            loss    = F.cross_entropy(output,target)
            pred    = output.max(1,keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().sum()

            losses.update(loss.item(),data.size(0))
            top1.update(correct.item()/data.size(0), data.size(0))
            
            if test_acc_every_batch:
                
                f.write('\nAccuracy: {}/{}({:.4f})'
                    .format(
                    correct.item(),
                    data.size(0),
                    top1.avg
                    )
                )
        
        temp1 = []
        for value in model.module.threshold.values():
            temp1 = temp1+[value.item()]    
        
        if epoch>5 and top1.avg<0.15:
            f.write('\n Quitting as the training is not progressing')
            exit(0)

        if top1.avg>max_accuracy:
            max_accuracy = top1.avg

            state = {
                    'accuracy'              : max_accuracy,
                    'epoch'                 : epoch,
                    'state_dict'            : model.state_dict(),
                    'optimizer'             : optimizer.state_dict(),
                    'thresholds'            : temp1,
                    'timesteps'             : timesteps,
                    'leak'                  : leak,
                    'activation'            : activation
                }
            try:
                os.mkdir('./trained_models/snn_ans/')
            except OSError:
                pass 
            filename = './trained_models/snn_ans/'+snn_identifier+'.pth'
            torch.save(state,filename)    
        
            #if is_best:
            #    shutil.copyfile(filename, 'best_'+filename)

        f.write(' test_loss: {:.4f}, test_acc: {:.4f}, best: {:.4f} time: {}'
            .format(
            losses.avg, 
            top1.avg,
            max_accuracy,
            datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            )
        )

def calculate_ans(adv_activations, epsilons):
    num_layers = len(adv_activations[0])
    ans_eps = []
    for eps in range(1,len(epsilons)):
        ans=[]
        #print(eps)
        for i in range(num_layers):
            a_clean = adv_activations[0][i] # epsilon 0 is clean
            a_adv = adv_activations[eps][i] # absolute val of a_adv
            Ans = torch.sum((a_clean-a_adv)**2, 1).sqrt()
            ans.append(torch.mean(Ans))
            del Ans
        ans_eps.append(ans)
        del ans
    return ans_eps

def plot_ans(ans_eps, epsilons, ann_identifier):
    num_layers = len(ans_eps[0])
    ans_eps_data = [list(x) for x in zip(epsilons[1:], ans_eps)]
    plt.figure(figsize=(10,10))
    for e in range(0, len(ans_eps_data)):
        plt.plot(range(0, num_layers), ans_eps_data[e][1], "*-", label='e = {}'.format(ans_eps_data[e][0]))
    plt.yticks(np.arange(0, max(ans_eps[-1]).item(), step=1))
    plt.xticks(np.arange(0, num_layers, step=1))
    plt.title("Adversarial Noise Sensitivity Ratio by Layer")
    plt.xlabel("Layers")
    plt.ylabel("ANS")
    plt.legend()
    plt.savefig('./logs/snn_ans/'+ann_identifier+'.png', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SNN training')
    parser.add_argument('--gpu',                    default=True,               type=bool,      help='use gpu')
    parser.add_argument('-s','--seed',              default=0,                  type=int,       help='seed for random number')
    parser.add_argument('--dataset',                default='CIFAR10',          type=str,       help='dataset name', choices=['MNIST','CIFAR10','CIFAR100'])
    parser.add_argument('--batch_size',             default=64,                 type=int,       help='minibatch size')
    parser.add_argument('-a','--architecture',      default='VGG16',            type=str,       help='network architecture', choices=['VGG5','VGG9','VGG11','VGG13','VGG16','VGG19'])
    parser.add_argument('-lr','--learning_rate',    default=1e-4,               type=float,     help='initial learning_rate')
    parser.add_argument('--pretrained_ann',         default='',                 type=str,       help='pretrained ANN model')
    parser.add_argument('--pretrained_snn',         default='',                 type=str,       help='pretrained SNN for inference')
    parser.add_argument('--test_only',              action='store_true',                        help='perform only inference')
    parser.add_argument('--log',                    action='store_true',                        help='to print the output on terminal or to log file')
    parser.add_argument('--epochs',                 default=300,                type=int,       help='number of training epochs')
    parser.add_argument('--lr_interval',            default='0.60 0.80 0.90',   type=str,       help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce',              default=10,                 type=int,       help='reduction factor for learning rate')
    parser.add_argument('--timesteps',              default=100,                type=int,       help='simulation timesteps')
    parser.add_argument('--leak',                   default=1.0,                type=float,     help='membrane leak')
    parser.add_argument('--scaling_factor',         default=0.7,                type=float,     help='scaling factor for thresholds at reduced timesteps')
    parser.add_argument('--default_threshold',      default=1.0,                type=float,     help='intial threshold to train SNN from scratch')
    parser.add_argument('--activation',             default='Linear',           type=str,       help='SNN activation function', choices=['Linear', 'STDB'])
    parser.add_argument('--alpha',                  default=0.3,                type=float,     help='parameter alpha for STDB')
    parser.add_argument('--beta',                   default=0.01,               type=float,     help='parameter beta for STDB')
    parser.add_argument('--optimizer',              default='Adam',             type=str,       help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])
    parser.add_argument('--weight_decay',           default=5e-4,               type=float,     help='weight decay parameter for the optimizer')    
    parser.add_argument('--momentum',               default=0.95,                type=float,     help='momentum parameter for the SGD optimizer')    
    parser.add_argument('--amsgrad',                default=True,               type=bool,      help='amsgrad parameter for Adam optimizer')
    parser.add_argument('--dropout',                default=0.3,                type=float,     help='dropout percentage for conv layers')
    parser.add_argument('--kernel_size',            default=3,                  type=int,       help='filter size for the conv layers')
    parser.add_argument('--test_acc_every_batch',   action='store_true',                        help='print acc of every batch during inference')
    parser.add_argument('--train_acc_batches',      default=200,                type=int,       help='print training progress after this many batches')
    parser.add_argument('--devices',                default='0',                type=str,       help='list of gpu device(s)')
    parser.add_argument('--ans',                    default=False,              type=bool,       help='find adversarially sensitive layers')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    
    # Seed random number
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    dataset             = args.dataset
    batch_size          = args.batch_size
    architecture        = args.architecture
    learning_rate       = args.learning_rate
    pretrained_ann      = args.pretrained_ann
    pretrained_snn      = args.pretrained_snn
    epochs              = args.epochs
    lr_reduce           = args.lr_reduce
    timesteps           = args.timesteps
    leak                = args.leak
    scaling_factor      = args.scaling_factor
    default_threshold   = args.default_threshold
    activation          = args.activation
    alpha               = args.alpha
    beta                = args.beta  
    optimizer           = args.optimizer
    weight_decay        = args.weight_decay
    momentum            = args.momentum
    amsgrad             = args.amsgrad
    dropout             = args.dropout
    kernel_size         = args.kernel_size
    test_acc_every_batch= args.test_acc_every_batch
    train_acc_batches   = args.train_acc_batches
    calcANS             = args.ans

    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))

    log_file = './logs/snn_ans/'
    try:
        os.mkdir(log_file)
    except OSError:
        pass 

    #snn_identifier = 'snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)+'_'+str(datetime.datetime.now())

    # get current time
    now = datetime.datetime.now() # current date and time
    date_time = now.strftime('%m-%d-%Y_%H-%M-%S')

    ann_identifier = date_time+'_ann_'+architecture.lower()+'_'+dataset.lower()
    snn_identifier = date_time+'_snn_'+architecture.lower()+'_'+dataset.lower()+'_'+str(timesteps)
    log_file+=ann_identifier+'.log'
    
    if args.log:
        f = open(log_file, 'w', buffering=1)
    else:
        f = sys.stdout

    if not pretrained_ann:
        ann_file = './trained_models/snn_ans/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'
        if os.path.exists(ann_file):
            val = input('\n Do you want to use the pretrained ANN {}? Y or N: '.format(ann_file))
            if val.lower()=='y' or val.lower()=='yes':
                pretrained_ann = ann_file
        else:
            ann_file = './trained_models/ann/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'
            if os.path.exists(ann_file):
                val = input('\n Do you want to use the pretrained ANN {}? Y or N: '.format(ann_file))
                if val.lower()=='y' or val.lower()=='yes':
                    pretrained_ann = ann_file

    f.write('\n Run on time: {}'.format(now))

    f.write('\n\n Arguments: ')
    for arg in vars(args):
        if arg == 'lr_interval':
            f.write('\n\t {:20} : {}'.format(arg, lr_interval))
        elif arg == 'pretrained_ann':
            f.write('\n\t {:20} : {}'.format(arg, pretrained_ann))
        else:
            f.write('\n\t {:20} : {}'.format(arg, getattr(args,arg)))
    f.write('\n')
    
    # Training settings
    if torch.cuda.is_available() and args.gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    adv_sens_layers = []
    state = torch.load(pretrained_ann, map_location='cpu')
    #If adv_sens_layers present in loaded ANN file
    if 'adv_sens_layers' in state.keys() and calcANS == False:
        adv_sens_layers = state['adv_sens_layers']
        f.write('\n Info: adv_sens_layers loaded from trained ANN: {}'.format(adv_sens_layers))
    else:

        f.write('\n Info: adv_sens_layers not found in trained ANN')
        f.write('\n Info: generating adv_sens_layers')

        """ LOAD ANN """
        
        f.write('\n*Loading ANN*\n')

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
        # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
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
        
        #CIFAR100 sometimes has problem to start training
        #One solution is to train for CIFAR10 with same architecture
        #Load the CIFAR10 trained model except the final layer weights
        model = nn.DataParallel(model)
        if pretrained_ann:
            cur_dict = model.state_dict()
            # print(cur_dict)
            for key in state['state_dict'].keys():
                curKey = key
                if 'module' not in curKey:
                    curKey = 'module.' + key
                if curKey in cur_dict:
                    if (state['state_dict'][key].shape == cur_dict[curKey].shape):
                        cur_dict[curKey] = nn.Parameter(state['state_dict'][key].data)
                        f.write('\n Success [ANN]: Loaded {} from {}'.format(key, pretrained_ann))
                    else:
                        f.write('\n Error [ANN]: Size mismatch at {}, size of loaded model {}, size of current model {}'.format(key, state['state_dict'][key].shape, cur_dict[curKey].shape))
                else:
                    f.write('\n Error [ANN]: Loaded weight {} not present in current model'.format(key))
            
            model.load_state_dict(cur_dict)
        
        f.write('\n ANN: {}'.format(model)) 

        if torch.cuda.is_available() and args.gpu:
            model.cuda()

        f.write('\n*Finished loading ANN*')


        """ ATTACK ANN """

        f.write('\n*Attacking ANN*')

        epsilons = [0, .5]
        f.write('\n {}'.format(epsilons))
        device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")

        # Run test for each epsilon
        adv_activations = [] # activations of last batch of each epsilon*** batch size of 64
        for eps in epsilons:
            adv_acc, adv_acts = testFGSM(test_loader_fgsm, model, device, eps)
            adv_activations.append(adv_acts) # activations of last batch
        f.write('\n Sets of activations: {}'.format(len(adv_activations)))

        f.write('\n*Finished attacking ANN*')

        
        """ GET ANS """

        f.write('\n*Getting ANS*')

        ans_eps = calculate_ans(adv_activations, epsilons)
        ans_eps_items = []
        for ans in ans_eps[0]:
            ans_eps_items.append(ans.item())
        f.write('\n ANS values:\n {}'.format(ans_eps_items))
        plot_ans(ans_eps, epsilons, ann_identifier)

        f.write('\n*Finished getting ANS*')


        """ FILTER ANS """

        f.write('\n*Filtering ANS*')

        threshold = 2
        for i in range(len(ans_eps_items)):
            if ans_eps_items[i] >= threshold:
                adv_sens_layers.append(i)
        f.write('\n Layers above threshold({}):\n {}'.format(threshold, adv_sens_layers))

        count = 0
        layers = model.module.cfg
        for l in layers:
            if isinstance(l, int):
                count += 1
        layers = []
        for i in range(count):
            layers.append(i*3)
        f.write('\n Actual layers in model:\n {}'.format(layers))
        temp = []
        for i in range(len(adv_sens_layers)):
            if adv_sens_layers[i] in layers:
                temp.append(adv_sens_layers[i])
        adv_sens_layers = temp
        f.write('\n Adversarially sensitive layers:\n {}'.format(adv_sens_layers))

        f.write('\n*Finished filtering ANS*')


        """ SAVING ANS """

        f.write('\n*Saving ANS*')

        try:
            os.mkdir('./trained_models/snn_ans/')
        except OSError:
            pass

        #Save the adv_sens_layers in the ANN file
        temp = {}
        pretrained_ann = './trained_models/snn_ans/ann_'+architecture.lower()+'_'+dataset.lower()+'.pth'
        for key,value in state.items():
            temp[key] = value
        temp['adv_sens_layers'] = adv_sens_layers
        torch.save(temp, pretrained_ann)
        f.write('\n Info: adv_sens_layers saved to trained ANN')

        f.write('\n*Finished saving ANS*')
    
    
    """ CONVERT TO SNN """

    f.write('\n*Converting to SNN*')

    normalize = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    
    if dataset in ['CIFAR10', 'CIFAR100']:
        transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])
        transform_test  = transforms.Compose([transforms.ToTensor(), normalize])

    if dataset == 'CIFAR10':
        trainset    = datasets.CIFAR10(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR10(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 10
    
    elif dataset == 'CIFAR100':
        trainset    = datasets.CIFAR100(root = '~/Datasets/cifar_data', train = True, download = True, transform = transform_train)
        testset     = datasets.CIFAR100(root='~/Datasets/cifar_data', train=False, download=True, transform = transform_test)
        labels      = 100
    
    elif dataset == 'MNIST':
        trainset   = datasets.MNIST(root='~/Datasets/mnist/', train=True, download=True, transform=transforms.ToTensor()
            )
        testset    = datasets.MNIST(root='~/Datasets/mnist/', train=False, download=True, transform=transforms.ToTensor())
        labels = 10

    elif dataset == 'IMAGENET':
        labels      = 1000
        traindir    = os.path.join('/local/scratch/a/imagenet/imagenet2012/', 'train')
        valdir      = os.path.join('/local/scratch/a/imagenet/imagenet2012/', 'val')
        trainset    = datasets.ImageFolder(
                            traindir,
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))
        testset     = datasets.ImageFolder(
                            valdir,
                            transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ])) 

    train_loader    = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader     = DataLoader(testset, batch_size=batch_size, shuffle=False)

    if architecture[0:3].lower() == 'vgg':
        model = VGG_SNN_STDB(vgg_name = architecture, activation = activation, labels=labels, timesteps=timesteps, leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout, kernel_size=kernel_size, dataset=dataset)
    
    elif architecture[0:3].lower() == 'res':
        model = RESNET_SNN_STDB(resnet_name = architecture, activation = activation, labels=labels, timesteps=timesteps,leak=leak, default_threshold=default_threshold, alpha=alpha, beta=beta, dropout=dropout, dataset=dataset)

    # if freeze_conv:
    #     for param in model.features.parameters():
    #         param.requires_grad = False
    
    #Please comment this line if you find key mismatch error and uncomment the DataParallel after the if block
    model = nn.DataParallel(model) 
    
    if pretrained_ann:
        state = torch.load(pretrained_ann, map_location='cpu')
        cur_dict = model.state_dict()     
        for key in state['state_dict'].keys():
            curKey = key
            if 'module' not in curKey:
                curKey = 'module.' + key
            if curKey in cur_dict:
                if (state['state_dict'][key].shape == cur_dict[curKey].shape):
                    cur_dict[curKey] = nn.Parameter(state['state_dict'][key].data)
                    f.write('\n Success: Loaded {} from {}'.format(key, pretrained_ann))
                else:
                    f.write('\n Error: Size mismatch at {}, size of loaded model {}, size of current model {}'.format(key, state['state_dict'][key].shape, cur_dict[curKey].shape))
            else:
                f.write('\n Error: Loaded weight {} not present in current model'.format(key))
        model.load_state_dict(cur_dict)
        f.write('\n Info: Accuracy of loaded ANN model: {}'.format(state['accuracy']))

        #If thresholds present in loaded ANN file
        if 'thresholds' in state.keys():
            thresholds = state['thresholds']
            f.write('\n Info: Thresholds loaded from trained ANN: {}'.format(thresholds))
            model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
        else:
            thresholds = find_threshold(batch_size=512, timesteps=1000, architecture=architecture)
            model.module.threshold_update(scaling_factor = scaling_factor, thresholds=thresholds[:])
            
            #Save the threhsolds in the ANN file
            temp = {}
            for key,value in state.items():
                temp[key] = value
            temp['thresholds'] = thresholds
            torch.save(temp, pretrained_ann)

    f.write('\n {}'.format(model))
    
    #model = nn.DataParallel(model) 
    if torch.cuda.is_available() and args.gpu:
        model.cuda()
    
    # ans_features = [12, 30]
    # ans_classifier = [0, 3, 6]
    f.write('\n {}'.format(adv_sens_layers))

    for parameter in model.parameters():
        parameter.requires_grad = False
    for i in range(len(model.module.features)):
        if i in adv_sens_layers:
            for parameter in model.module.features[i].parameters():
                parameter.requires_grad = True
    # for i in range(len(model.module.classifier)):
    #     if i in ans_classifier:
    #         for parameter in model.module.classifier[i].parameters():
    #             parameter.requires_grad = True
    
    # for name, param in model.state_dict().items():
    #     f.write('\n{}'.format(name))

    for name, param in model.named_parameters():
        if param.requires_grad:
            f.write('\n{}'.format(name))

    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    
    f.write('\n {}'.format(optimizer))
    max_accuracy = 0
    
    #print(model)
    #f.write('\n Threshold: {}'.format(model.module.threshold))

    for epoch in range(1, epochs):
        start_time = datetime.datetime.now()
        if not args.test_only:
            train(epoch)
        test(epoch)

    f.write('\n Highest accuracy: {:.4f}'.format(max_accuracy))
    f.write('\n Total script time: {}'.format(datetime.timedelta(seconds=(datetime.datetime.now() - now).seconds)))
    f.close()