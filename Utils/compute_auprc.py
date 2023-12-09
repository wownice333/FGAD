from sklearn.metrics import auc, precision_recall_curve
def compute_auprc_baseline(target, score):
    precision, recall, thresholds = precision_recall_curve(target, -1*score)
    auprc = auc(recall, precision)
    return auprc

def compute_auprc(target, score):
    precision, recall, thresholds = precision_recall_curve(target, score)
    auprc = auc(recall, precision)
    return auprc