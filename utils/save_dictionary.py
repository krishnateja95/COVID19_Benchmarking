from utils.checkpoint import save_checkpoint

def store_dict(args, avg_train_time, loss, top1, f1, precision, recall, FPR, FNR, MCC, epoch, save_dict, 
               model_state_dict, optimizer, save_ckpt, save_best = False):
    
    if loss < save_dict['best_loss']:
        save_dict['best_loss']       = loss
        save_dict['best_loss_epoch'] = epoch
        
        if save_ckpt: 
            save_checkpoint(args, epoch, avg_train_time, model_state_dict, loss, top1, f1, precision, recall,
                            FPR, FNR, MCC, optimizer, False, save_best, 'loss')
        
    if top1 > save_dict['best_top1']:
        save_dict['best_top1']       = top1
        save_dict['best_top1_epoch'] = epoch
        
        if save_ckpt:
            save_checkpoint(args, epoch, avg_train_time, model_state_dict, loss, top1, f1, precision, recall,
                            FPR, FNR, MCC, optimizer, False, save_best, 'top1')

    if f1 > save_dict['best_f1']:
        save_dict['best_f1']       = f1
        save_dict['best_f1_epoch'] = epoch    
        
        if save_ckpt:
            save_checkpoint(args, epoch, avg_train_time, model_state_dict, loss, top1, f1, precision, recall,
                            FPR, FNR, MCC, optimizer, False, save_best, 'f1')
        
    if precision > save_dict['best_precision']:
        save_dict['best_precision']       = precision
        save_dict['best_precision_epoch'] = epoch
        
        if save_ckpt:
            save_checkpoint(args, epoch, avg_train_time, model_state_dict, loss, top1, f1, precision, recall,
                            FPR, FNR, MCC, optimizer, False, save_best, 'precision')
        
    if recall > save_dict['best_recall']:
        save_dict['best_recall']       = recall
        save_dict['best_recall_epoch'] = epoch

        if save_ckpt:
            save_checkpoint(args, epoch, avg_train_time, model_state_dict, loss, top1, f1, precision, recall,
                            FPR, FNR, MCC, optimizer, False, save_best, 'recall')
        
    if FPR < save_dict['best_FPR']:
        save_dict['best_FPR']       = FPR
        save_dict['best_FPR_epoch'] = epoch    
        
        if save_ckpt:
            save_checkpoint(args, epoch, avg_train_time, model_state_dict, loss, top1, f1, precision, recall,
                            FPR, FNR, MCC, optimizer, False, save_best, 'FPR')
        
    if FNR < save_dict['best_FNR']:
        save_dict['best_FNR']       = FNR
        save_dict['best_FNR_epoch'] = epoch          
        
        if save_ckpt:
            save_checkpoint(args, epoch, avg_train_time, model_state_dict, loss, top1, f1, precision, recall,
                            FPR, FNR, MCC, optimizer, False, save_best, 'FNR')
        
    if MCC > save_dict['best_MCC']:
        save_dict['best_MCC']       = MCC
        save_dict['best_MCC_epoch'] = epoch
        
        if save_ckpt:
            save_checkpoint(args, epoch, avg_train_time, model_state_dict, loss, top1, f1, precision, recall,
                            FPR, FNR, MCC, optimizer, False, save_best, 'MCC')
    
    return save_dict

