import os
import torch

def save_checkpoint(args, epoch, avg_train_time, model_state_dict, loss, top1, f1, precision, recall, 
                    FPR, FNR, MCC, optimizer, save_current, save_best, best_file_name):
    save_dict = {
        'epoch': epoch,
        'avg_train_time': avg_train_time,
        'model_name': args.model_name,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'top1': top1,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'FPR': FPR,
        'FNR': FNR,
        'MCC': MCC
        }
    
    if save_current:
        current_model = args.model_dir + str(args.model_type) + '/Current_Models/' + str(args.model_name) + '_current.pt'
        torch.save(save_dict, current_model)
    
    if save_best:
        best_model = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_' + best_file_name + '.pt'
        torch.save(save_dict, best_model)
    
    
def load_state(args, model, optimizer, save_dict, best_metric, evaluate=False):
    
    best_PATH_loss      = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_loss.pt'
    best_PATH_top1      = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_top1.pt'
    best_PATH_f1        = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_f1.pt'
    best_PATH_precision = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_precision.pt'
    best_PATH_recall    = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_recall.pt'
    best_PATH_FPR       = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_FPR.pt'
    best_PATH_FNR       = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_FNR.pt'
    best_PATH_MCC       = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_MCC.pt'

    current_PATH = args.model_dir + str(args.model_type) + '/Current_Models/' + str(args.model_name) + '_current.pt'
        
    if os.path.isfile(current_PATH):
        print("Checkpoint found")
        
        current_checkpoint = torch.load(current_PATH)
        model.load_state_dict(current_checkpoint['model_state_dict'])
        optimizer.load_state_dict(current_checkpoint['optimizer_state_dict'])
        
        epoch                               =  current_checkpoint['epoch'] + 1
        avg_train_time                      =  current_checkpoint['avg_train_time']
        
        
        if os.path.isfile(best_PATH_loss):
            loss_checkpoint                     =  torch.load(best_PATH_loss)
            save_dict['best_loss']              =  loss_checkpoint['loss']
            save_dict['best_loss_epoch']        =  loss_checkpoint['epoch']
        
        
        
        if os.path.isfile(best_PATH_top1):
            top1_checkpoint                     =  torch.load(best_PATH_top1)
            save_dict['best_top1']              =  top1_checkpoint['top1']
            save_dict['best_top1_epoch']        =  top1_checkpoint['epoch']
        
        
        if os.path.isfile(best_PATH_f1):
            f1_checkpoint                       =  torch.load(best_PATH_f1)
            save_dict['best_f1']                =  f1_checkpoint['f1_score']
            save_dict['best_f1_epoch']          =  f1_checkpoint['epoch']
        
        
        if os.path.isfile(best_PATH_precision):
            precision_checkpoint                =  torch.load(best_PATH_precision)
            save_dict['best_precision']         =  precision_checkpoint['precision']
            save_dict['best_precision_epoch']   =  precision_checkpoint['epoch']

        
        if os.path.isfile(best_PATH_recall):
            recall_checkpoint                   =  torch.load(best_PATH_recall)
            save_dict['best_recall']            =  recall_checkpoint['recall']
            save_dict['best_recall_epoch']      =  recall_checkpoint['epoch']

        
        if os.path.isfile(best_PATH_FPR):
            FPR_checkpoint                      =  torch.load(best_PATH_FPR)
            save_dict['best_FPR']               =  FPR_checkpoint['FPR']
            save_dict['best_FPR_epoch']         =  FPR_checkpoint['epoch']

        
        if os.path.isfile(best_PATH_FNR):
            FNR_checkpoint                      =  torch.load(best_PATH_FNR)
            save_dict['best_FNR']               =  FNR_checkpoint['FNR']
            save_dict['best_FNR_epoch']         =  FNR_checkpoint['epoch']

        
        if os.path.isfile(best_PATH_MCC):
            MCC_checkpoint                      =  torch.load(best_PATH_MCC)
            save_dict['best_MCC']               =  MCC_checkpoint['MCC']
            save_dict['best_MCC_epoch']         =  MCC_checkpoint['epoch']

    else:
        print("No Checkpoint found")
        epoch, avg_train_time  = 0, 0
    
    
    if evaluate:
        best_model_PATH = args.model_dir + str(args.model_type) + '/Best_Models/' + str(args.model_name) + '_best_' + best_metric + '.pt'
        best_checkpoint = torch.load(best_model_PATH)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
    
    return model, optimizer, save_dict, epoch, avg_train_time 
    
    
    