import sys
import datetime
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import json
import warnings
warnings.filterwarnings("ignore")

# Your existing code

def optimize_hyperparameter(eta_candid, max_depth_candid, num_round_candid):
    best_f1 = 0
    best_params = {}
    for eta in eta_candid:
        for max_depth in max_depth_candid:
            for num_round in num_round_candid:
                print(eta, max_depth, num_round)
                params["eta"] = eta
                params["max_depth"] = max_depth
                precision_list, recall_list, f1_list, _, _ = train_loop(num_round)
                mean_f1 = np.mean(f1_list)
                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_params = {
                        'eta': eta,
                        'max_depth': max_depth,
                        'num_round': num_round
                    }
    return best_params, best_f1

def train_loop(num_round=300):
    # Your existing training code
    pass

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--tune':
        eta_candidate = [0.08, 0.05, 0.03, 0.01]
        max_depth_candidate = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
        num_round_candidate = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        best_params, best_f1 = optimize_hyperparameter(eta_candidate, max_depth_candidate, num_round_candidate)
        print(json.dumps({'best_params': best_params, 'best_f1': best_f1}))
        sys.exit()

    if len(sys.argv) > 1 and sys.argv[1] == '--params':
        if len(sys.argv) != 5:
            print("Usage: python train.py --params <max_depth> <eta> <num_round>")
            sys.exit(1)
        max_depth = int(sys.argv[2])
        eta = float(sys.argv[3])
        num_round = int(sys.argv[4])
        params = {
            'max_depth': max_depth,
            'eta': eta,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
        }
        precision_list, recall_list, f1_list, _, _ = train_loop(num_round)
        print("Average F1: %.3f" % np.mean(f1_list))
