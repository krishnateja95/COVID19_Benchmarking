from utils.validation_metrics import accuracy, f1_score_one_hot, precision_one_hot,recall_one_hot
from utils.validation_metrics import false_positive_rate_one_hot, false_negative_rate_one_hot, mcc_one_hot

def get_all_validation_metrics(output, target, target_var, target_unsqueeze, criterion, loss_counter, top1_counter,
                               f1_counter, precision_counter, recall_counter, FPR_counter, FNR_counter, MCC_counter, input_data):
    
    #loss
    loss = criterion(output, target_var)
    loss_counter.update(loss.item(), input_data.size(0))
    
    #top1 accuracy
    prec1, prec3 = accuracy(output, target_unsqueeze)
    top1_counter.update(prec1.item(),  input_data.size(0))
    
    #f1score
    f1_score = f1_score_one_hot(output, target)
    f1_counter.update(f1_score, input_data.size(0))

    #precision
    precision_per_batch = precision_one_hot(output, target)
    precision_counter.update(precision_per_batch, input_data.size(0))

    #recall
    recall_per_batch = recall_one_hot(output, target)
    recall_counter.update(recall_per_batch, input_data.size(0))

    #FPR
    FPR_per_batch = false_positive_rate_one_hot(output, target)
    FPR_counter.update(FPR_per_batch, input_data.size(0))


    #FNR
    FNR_per_batch = false_negative_rate_one_hot(output, target)
    FNR_counter.update(FNR_per_batch, input_data.size(0))

    #MCC
    MCC_per_batch = mcc_one_hot(output, target)
    MCC_counter.update(MCC_per_batch, input_data.size(0))
    
    return loss, loss_counter, top1_counter, f1_counter, precision_counter, recall_counter, FPR_counter, FNR_counter, MCC_counter


