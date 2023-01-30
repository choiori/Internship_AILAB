import sklearn.metrics as skmet
import numpy as np
import os 
from terminaltables import SingleTable
from termcolor import colored


def result_metric(seq, y_trues, y_preds, loss_type):

    
    y_pred_argmax = np.argmax(y_preds, 1)
    result_dict = skmet.classification_report(y_trues, y_pred_argmax, digits=3, output_dict=True)
    cm = skmet.confusion_matrix(y_trues, y_pred_argmax)
    
    accuracy = round(result_dict['accuracy']*100, 1)
    macro_f1 = round(result_dict['macro avg']['f1-score']*100, 1)
    kappa = round(skmet.cohen_kappa_score(y_trues, y_pred_argmax), 3)
    
    wpr = round(result_dict['0.0']['precision']*100, 1)
    wre = round(result_dict['0.0']['recall']*100, 1)
    wf1 = round(result_dict['0.0']['f1-score']*100, 1)
    
    n1pr = round(result_dict['1.0']['precision']*100, 1)
    n1re = round(result_dict['1.0']['recall']*100, 1)
    n1f1 = round(result_dict['1.0']['f1-score']*100, 1)

    n2pr = round(result_dict['2.0']['precision']*100, 1)
    n2re = round(result_dict['2.0']['recall']*100, 1)
    n2f1 = round(result_dict['2.0']['f1-score']*100, 1)
    
    n3pr = round(result_dict['3.0']['precision']*100, 1)
    n3re = round(result_dict['3.0']['recall']*100, 1)
    n3f1 = round(result_dict['3.0']['f1-score']*100, 1)
    
    rpr = round(result_dict['4.0']['precision']*100, 1)
    rre = round(result_dict['4.0']['recall']*100, 1)
    rf1 = round(result_dict['4.0']['f1-score']*100, 1)
    
    
    overall_data = [
        ['ACC', 'MF1', '\u03BA'],
        [accuracy, macro_f1, kappa],
    ]
    
    perclass_data = [
        [colored('A', 'cyan') + '\\' + colored('P', 'green'), 'W', 'N1', 'N2', 'N3', 'R', 'PR', 'RE', 'F1'],
        ['W', cm[0][0], cm[0][1], cm[0][2], cm[0][3], cm[0][4], wpr, wre, wf1],
        ['N1', cm[1][0], cm[1][1], cm[1][2], cm[1][3], cm[1][4], n1pr, n1re, n1f1],
        ['N2', cm[2][0], cm[2][1], cm[2][2], cm[2][3], cm[2][4], n2pr, n2re, n2f1],
        ['N3', cm[3][0], cm[3][1], cm[3][2], cm[3][3], cm[3][4], n3pr, n3re, n3f1],
        ['R', cm[4][0], cm[4][1], cm[4][2], cm[4][3], cm[4][4], rpr, rre, rf1],
    ]
    
    
    overall_dt = SingleTable(overall_data, colored('OVERALL RESULT', 'red'))
    perclass_dt = SingleTable(perclass_data, colored('PER-CLASS RESULT', 'red'))
        
    
    print(f'\n\n[INFO] sequence_length : {seq}, Evaluation result from fold 1 to 20' )
    print('\n' + overall_dt.table)
    print('\n' + perclass_dt.table)
    print(colored(' A', 'cyan') + ': Actual Class, ' + colored('P', 'green') + ': Predicted Class' + '\n\n')

    
