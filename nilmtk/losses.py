from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, precision_score, matthews_corrcoef
import numpy as np
on_threhold = {'fridge':50, 'kettle':2000, 'dish washer':10, 'washing machine':20, 'drill':0}

def mae(app_name,app_gt,app_pred):
    return mean_absolute_error(app_gt,app_pred)

def rmae(app_name,app_gt,app_pred):
    constant = 1
    numerator = np.abs(app_gt - app_pred)
    max_temp = np.where(app_gt>app_pred,app_gt,app_pred)
    denominator = constant + max_temp
    return np.mean(numerator/denominator)

def nep(app_name,app_gt,app_pred):
    numerator = np.sum(np.abs(app_gt - app_pred))
    denominator = np.sum(np.abs(app_gt))
    return numerator/denominator

def rmse(app_name,app_gt, app_pred):
    return mean_squared_error(app_gt,app_pred)**(.5)

def recall(app_name,app_gt, app_pred):
    threshold = on_threhold.get(app_name,10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp<threshold,0,1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp<threshold,0,1)

    return recall_score(gt_temp, pred_temp)

def precision(app_name,app_gt, app_pred):
    threshold = on_threhold.get(app_name,10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp<threshold,0,1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp<threshold,0,1)

    return precision_score(gt_temp, pred_temp)

def f1score(app_name,app_gt, app_pred):
    threshold = on_threhold.get(app_name,10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp<threshold,0,1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp<threshold,0,1)

    return f1_score(gt_temp, pred_temp)

def MCC(app_name,app_gt, app_pred):
    threshold = on_threhold.get(app_name,10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp<threshold,0,1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp<threshold,0,1)

    return matthews_corrcoef(gt_temp, pred_temp)

def omae(app_name, app_gt, app_pred):
    threshold = on_threhold.get(app_name,10)
    gt_temp = np.array(app_gt)
    idx = gt_temp > threshold
    gt_temp = gt_temp[idx]
    pred_temp = np.array(app_pred)
    pred_temp = pred_temp[idx] 

    return mae(app_name, gt_temp, pred_temp)

def ormae(app_name,app_gt, app_pred):
    threshold = on_threhold.get(app_name,10)
    gt_temp = np.array(app_gt)
    idx = gt_temp > threshold
    gt_temp = gt_temp[idx]
    pred_temp = np.array(app_pred)
    pred_temp = pred_temp[idx]

    return rmae(gt_temp, pred_temp)



