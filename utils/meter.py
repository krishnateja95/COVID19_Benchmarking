




class AverageMeter_predictions(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, correct, total):
        self.correct += correct
        self.total += total
              
    def get_top1_acc(self):
        return 100*float(self.correct/self.total) 
    
    
    def f1_score(self):
        return
        
    


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
        
        
        
        
        
        
        
        
        
        