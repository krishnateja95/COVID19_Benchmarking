import os
import time
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import csv
import os.path
import timeit
import math
from thop import profile

sys.path.append('../')
sys.path.append('../../')
import warnings

warnings.filterwarnings("ignore")

from utils.get_model import get_neural_network

from utils.get_data import get_COVID10_dataloader
from utils.checkpoint import load_state, save_checkpoint
from utils.scheduler import LRScheduler

from utils.validation_metrics import AverageMeter
from utils.validation_metrics import calculate_new_average, f1_score_one_hot, precision_one_hot, recall_one_hot
from utils.validation_metrics import false_positive_rate_one_hot, false_negative_rate_one_hot, mcc_one_hot
from utils.performance_metrics import get_macs_params, get_flops, measure_latency
from utils.latency_measurement import measure_latency
from utils.get_all_metrics import get_all_validation_metrics
from utils.save_dictionary import store_dict
from utils.memory_utils import estimate_memory_training, estimate_memory_inference

parser = argparse.ArgumentParser(description='Pytorch COVID19 Training')


parser.add_argument('--model_name',   type=str, default = 'ResNet18')
parser.add_argument('--model_family', type=str, default = 'ResNet_family')
parser.add_argument('--model_type',   type=str, default = 'CNN_Models')

parser.add_argument('--using_bn', default=True, type=bool, help='Use batch normalization')

parser.add_argument('--port', default=29500, type=int, help='port of server')

parser.add_argument('--model_dir', type=str, default = '/work/arun/COVID19_research/train/model_checkpoints/')

parser.add_argument('--train_root', type=str, default = '/work/arun/COVID19_research/dataset/COVID_19_Classification/Train')
parser.add_argument('--test_root', type=str, default  = '/work/arun/COVID19_research/dataset/COVID_19_Classification/Test/')
parser.add_argument('--val_root', type=str, default   = '/work/arun/COVID19_research/dataset/COVID_19_Classification/Val')

parser.add_argument('--epochs', default=500, type=int, help='Number of Epochs')
parser.add_argument('--batch_size', default=256, type=int, help='Batch Size')
parser.add_argument('--workers', default=3, type=int, help='Number of Workers')

parser.add_argument('--lr_mode', default='cosine', type=str, help='Learning rate mode')
parser.add_argument('--base_lr', default=0.1, type=float, help='Base Learning Rate')
parser.add_argument('--warmup_epochs', default=25, type=int, help='Number of warmup epochs')
parser.add_argument('--warmup_lr', default=0.0, type=float, help='Warmup Learning Rate')
parser.add_argument('--targetlr', default=0.0, type=float, help='Target Learning Rate')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight_decay', default=0.00005, type=float, help='Weight Decay')
parser.add_argument('--using_moving_average', default=True, type=bool, help='Use moving Average')
parser.add_argument('--last_gamma', default=True, type=bool, help='Last Gamma')

parser.add_argument('--print_freq', default=100, type=int, help='Print frequency')

parser.add_argument('--resume_from', default='', help='resume_from')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
#args = parser.parse_args()

def main():
    global args, best_prec1, num_classes
    args = parser.parse_args()
    
    train_loader, test_loader, val_loader = get_COVID10_dataloader(args)
    
    num_classes = len(set(test_loader.dataset.targets))
    
    # create model
    print("=> creating model '{}'".format(args.model_name))

    model = get_neural_network(args.model_name)(num_classes = num_classes)
    
    CPU_latency  = measure_latency(model, device_type = 'cpu')
    macs, params = get_macs_params(model.to('cpu'))
    flops        = get_flops(model.to('cpu'))
    
    inference_memory = estimate_memory_inference(model)
    
    model.cuda()
    
    GPU_latency = measure_latency(model, device_type = 'gpu')
    training_memory  = estimate_memory_training(model, )
    
    criterion = nn.CrossEntropyLoss().cuda()
#     optimizer = torch.optim.Adam(model.parameters(), args.base_lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # auto resume from a checkpoint
    model_dir = args.model_dir
    
    start_epoch = 0
    
    
    save_dict = {'best_loss':            math.inf,
                 'best_loss_epoch':      0,
                 'best_top1':            -math.inf,
                 'best_top1_epoch':      0,
                 'best_f1':              -math.inf,
                 'best_f1_epoch':        0,    
                 'best_precision':       -math.inf,
                 'best_precision_epoch': 0,
                 'best_recall':          -math.inf,
                 'best_recall_epoch':    0,
                 'best_FPR':             math.inf,
                 'best_FPR_epoch':       0,    
                 'best_FNR':             math.inf,
                 'best_FNR_epoch':       0,          
                 'best_MCC':             -math.inf,
                 'best_MCC_epoch':       0
                 }       
    
    if args.evaluate:
        model, optimizer, save_dict, start_epoch, avg_train_time = load_state(args, model, optimizer, save_dict, 'top1', evaluate=True)
    else:
        model, optimizer, save_dict, start_epoch, avg_train_time = load_state(args, model, optimizer, save_dict, None, evaluate=False)
    
    print("start_epoch", start_epoch)
    
    cudnn.benchmark = True
    
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    #niters = len(train_loader)
    niters = len(train_loader.dataset) // args.batch_size

    lr_scheduler = LRScheduler(optimizer, niters, args)

    train_save_dict = save_dict.copy()
    test_save_dict  = save_dict.copy() 
    val_save_dict   = save_dict.copy() 
    
    for epoch in range(start_epoch, args.epochs):
        print("epoch", epoch)
        
        start_time = timeit.default_timer()
        
        train_loss, train_top1, train_f1, train_precision,\
        train_recall, train_FPR, train_FNR, train_MCC = train(train_loader, model, criterion, optimizer, lr_scheduler, epoch)
        
        end_time = timeit.default_timer()
        current_train_time = end_time - start_time
        avg_train_time = calculate_new_average(avg_train_time, epoch, current_train_time)
        
        train_save_dict = store_dict(args, avg_train_time, train_loss, train_top1, train_f1, train_precision, train_recall,
                                     train_FPR, train_FNR, train_MCC ,epoch, train_save_dict, model.state_dict(), 
                                     optimizer, False, True).copy()
        
        # evaluate on validation set
        val_loss, val_top1, val_f1, val_precision, \
        val_recall, val_FPR, val_FNR, val_MCC = validate(val_loader, model, criterion, epoch)
        
        val_save_dict   = store_dict(args, avg_train_time, val_loss, val_top1, val_f1, val_precision, val_recall,
                                     val_FPR, val_FNR, val_MCC, epoch, val_save_dict, model.state_dict(), 
                                     optimizer, False, True).copy()
        
        # evaluate on test set
        test_loss, test_top1, test_f1, test_precision,\
        test_recall, test_FPR, test_FNR, test_MCC  = validate(test_loader, model, criterion, epoch)
        
        test_save_dict  = store_dict(args, avg_train_time, test_loss, test_top1, test_f1, test_precision, test_recall,
                                     test_FPR, test_FNR, test_MCC, epoch, test_save_dict, model.state_dict(), 
                                     optimizer, True, True).copy()
        
        save_checkpoint(args, epoch, avg_train_time, model.state_dict(), test_loss, test_top1, test_f1,
                        test_precision, test_recall, test_FPR, test_FNR, test_MCC, optimizer, True, False, None)
        
    
    train_save_dict['Train_Time']       = test_save_dict['Train_Time']       = val_save_dict['Train_Time']       = avg_train_time
    train_save_dict['Model_Name']       = test_save_dict['Model_Name']       = val_save_dict['Model_Name']       = args.model_name 
    train_save_dict['MACs']             = test_save_dict['MACs']             = val_save_dict['MACs']             = macs 
    train_save_dict['FLOPs']            = test_save_dict['FLOPs']            = val_save_dict['FLOPs']            = flops 
    train_save_dict['Number_of_Params'] = test_save_dict['Number_of_Params'] = val_save_dict['Number_of_Params'] = params
    train_save_dict['CPU_latency']      = test_save_dict['CPU_latency']      = val_save_dict['CPU_latency']      = CPU_latency 
    train_save_dict['GPU_latency']      = test_save_dict['GPU_latency']      = val_save_dict['GPU_latency']      = GPU_latency 
    train_save_dict['training_memory']  = test_save_dict['training_memory']  = val_save_dict['training_memory']  = training_memory 
    train_save_dict['inference_memory'] = test_save_dict['inference_memory'] = val_save_dict['inference_memory'] = inference_memory 
    
    
    export_dictionary_to_csv(train_save_dict,'results/', 'train_data.csv')
    export_dictionary_to_csv(train_save_dict,'results/' + str(args.model_type) + '/' + str(args.model_family) + '/', 'train_data.csv')
    
    export_dictionary_to_csv(val_save_dict, 'results/', 'val_data.csv')
    export_dictionary_to_csv(val_save_dict, 'results/' + str(args.model_type) + '/' + str(args.model_family) + '/', 'val_data.csv')
    
    export_dictionary_to_csv(test_save_dict,'results/', 'test_data.csv')
    export_dictionary_to_csv(test_save_dict,'results/' + str(args.model_type) + '/' + str(args.model_family) + '/', 'test_data.csv')
    
def export_dictionary_to_csv(dictionary, foldername, filename):
    
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    
    # Check if the file exists
    file_exists = os.path.isfile(foldername+filename)
    rows = []
    
    header_row = dictionary.keys()
    
    if not file_exists:
        rows.append(header_row)

    value_rows = []
    # Generate data rows
    for key, values in dictionary.items():
        value_rows.append(str(dictionary[key]))
    rows.append(value_rows)
    
    # Open the CSV file in append mode if it exists, else write mode
    with open(foldername + filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows) # Write rows
    
    print(f"Data successfully exported to {filename}.")
        
def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch):
    loss_counter      = AverageMeter()
    top1_counter      = AverageMeter()
    f1_counter        = AverageMeter()
    precision_counter = AverageMeter()
    recall_counter    = AverageMeter()
    FPR_counter       = AverageMeter()
    FNR_counter       = AverageMeter()
    MCC_counter       = AverageMeter()
    
    model.train()

    for i, (input_data, target) in enumerate(train_loader):
        input_var  = torch.autograd.Variable(input_data.cuda())
        lr_scheduler.update(i, epoch)
        
        target_unsqueeze = target.clone().detach()
        target_unsqueeze = target_unsqueeze.cuda(non_blocking=True)
        target_unsqueeze = torch.autograd.Variable(target_unsqueeze)
        
        target     = torch.zeros(target.size(0), num_classes).scatter_(1, target.unsqueeze(1), 1)
        target     = target.cuda(non_blocking=True)
        target_var = torch.autograd.Variable(target)
        
        output = model(input_var)
        
        loss, loss_counter, top1_counter, f1_counter, precision_counter, recall_counter, \
        FPR_counter, FNR_counter, MCC_counter = get_all_validation_metrics(output, target, target_var, target_unsqueeze, criterion,
                                                                           loss_counter,  top1_counter, f1_counter, precision_counter, 
                                                                           recall_counter, FPR_counter,  
                                                                           FNR_counter, MCC_counter, input_data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss       =  loss_counter.avg
    top1       =  top1_counter.avg
    Error      =  100 - top1 
    f1         =  f1_counter.avg
    precision  =  precision_counter.avg
    recall     =  recall_counter.avg
    FPR        =  FPR_counter.avg
    FNR        =  FNR_counter.avg
    MCC        =  MCC_counter.avg
    
    return loss, top1, f1, precision, recall, FPR, FNR, MCC
          
def validate(val_loader, model, criterion, epoch):
    loss_counter       = AverageMeter()
    top1_counter       = AverageMeter()
    f1_counter         = AverageMeter()
    precision_counter  = AverageMeter()
    recall_counter     = AverageMeter()
    FPR_counter        = AverageMeter()
    FNR_counter        = AverageMeter()
    MCC_counter        = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input_data, target) in enumerate(val_loader):
            
            input_var = torch.autograd.Variable(input_data.cuda())
            
            target_unsqueeze = target.clone().detach()
            target_unsqueeze = target_unsqueeze.cuda(non_blocking=True)
            target_unsqueeze = torch.autograd.Variable(target_unsqueeze)

            target     = torch.zeros(target.size(0), num_classes).scatter_(1, target.unsqueeze(1), 1)
            target     = target.cuda(non_blocking=True)
            target_var = torch.autograd.Variable(target)

            
            # compute output
            output = model(input_var)
            
            loss, loss_counter, top1_counter, f1_counter, precision_counter, recall_counter, \
            FPR_counter, FNR_counter, MCC_counter = get_all_validation_metrics(output, target, target_var, target_unsqueeze, 
                                                                               criterion, loss_counter, top1_counter, f1_counter,
                                                                               precision_counter, recall_counter,
                                                                               FPR_counter, FNR_counter, MCC_counter, input_data)
        niter = (epoch + 1)
    
    loss        =  loss_counter.avg
    top1        =  top1_counter.avg
    Error       =  100 - top1 
    f1          =  f1_counter.avg
    precision   =  precision_counter.avg
    recall      =  recall_counter.avg
    FPR         =  FPR_counter.avg
    FNR         =  FNR_counter.avg
    MCC         =  MCC_counter.avg

    return loss, top1, f1, precision, recall, FPR, FNR, MCC


if __name__ == '__main__':
    main()
