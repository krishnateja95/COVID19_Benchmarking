import torch
import numpy as np 
from sklearn.metrics import f1_score 
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

def get_num_predictions(output, target):
    _, predicted = torch.max(output, dim=1)
    correct = (predicted == torch.argmax(target, dim=1)).sum().item()
    return correct, target.size(0)

def calculate_new_average(previous_average, current_epoch, new_value):
    new_average = ((previous_average * current_epoch) + new_value) / (current_epoch + 1)
    return new_average


class AverageMeter(object):
    def __init__(self):
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


def accuracy(output, target, topk=(1,3)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)) 

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def f1_score_one_hot(output, target):
    y_true = torch.argmax(target, dim=1).cpu().numpy()
    y_pred = torch.argmax(output, dim=1).cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1

def precision_one_hot(output, target):
    y_true = torch.argmax(target, dim=1).cpu().numpy()
    y_pred = torch.argmax(output, dim=1).cpu().numpy()
    precision = precision_score(y_true, y_pred, average='weighted')
    return precision

 
def recall_one_hot(output, target):
    y_true = torch.argmax(target, dim=1).cpu().numpy()
    y_pred = torch.argmax(output, dim=1).cpu().numpy()
    recall = recall_score(y_true, y_pred, average='weighted')
    return recall


def false_positive_rate_one_hot(output, target):
    y_true = torch.argmax(target, dim=1).cpu().numpy()
    y_pred = torch.argmax(output, dim=1).cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)

    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (fp + np.diag(cm))
    fp_rate = fp / (fp + tn)

    fp_rate = fp_rate.mean()

    return fp_rate




def false_negative_rate_one_hot(output, target):
    y_true = torch.argmax(target, dim=1).cpu().numpy()
    y_pred = torch.argmax(output, dim=1).cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)

    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    fn_rate = fn / (fn + tp)

    fn_rate = fn_rate.mean()

    return fn_rate



def mcc_one_hot(output, target):
    y_true = torch.argmax(target, dim=1).cpu().numpy()
    y_pred = torch.argmax(output, dim=1).cpu().numpy()

    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc


