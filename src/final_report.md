=== rf ===
Training Classifier (rf) on target: crisis_7d
Optimized Decision Threshold: 0.35 (Max Train F1: 0.70)
Accuracy: 72.34%
ROC-AUC : 0.6007 (Ranking Ability)
F1-Score: 0.3810 (Balance Precision/Recall)

Classification Report:
              precision    recall  f1-score   support

           0       0.77      0.88      0.82        34
           1       0.50      0.31      0.38        13

    accuracy                           0.72        47
   macro avg       0.63      0.60      0.60        47
weighted avg       0.69      0.72      0.70        47


Feature Importances:
----------------------------------------
fuel_stress_index              : 0.2034
share_logistics_terms          : 0.1430
total_messages                 : 0.0936
count_logistics_terms          : 0.0910
unique_messages                : 0.0794
keyword_mentions               : 0.0690
price_sum_gasoline             : 0.0589
price_sum                      : 0.0511
price_sum_diesel               : 0.0498
price_mentions                 : 0.0451
avg_price                      : 0.0364
price_mentions_gasoline        : 0.0272
price_mentions_diesel          : 0.0251
count_gasoline_terms           : 0.0096
share_gasoline_terms           : 0.0070
share_diesel_terms             : 0.0048
count_diesel_terms             : 0.0039
share_queue_terms              : 0.0011
count_queue_terms              : 0.0002
count_infra_terms              : 0.0002
share_infra_terms              : 0.0000
count_policy_terms             : 0.0000
count_shortage_terms           : 0.0000
sentiment_mean                 : 0.0000
sentiment_min                  : 0.0000
share_agro_terms               : 0.0000
share_rationing_terms          : 0.0000
share_price_terms              : 0.0000
share_policy_terms             : 0.0000
count_price_terms              : 0.0000
share_shortage_terms           : 0.0000
count_rationing_terms          : 0.0000
count_agro_terms               : 0.0000
----------------------------------------
=== gb ===
Training Classifier (gb) on target: crisis_7d
Optimized Decision Threshold: 0.25 (Max Train F1: 0.75)
Accuracy: 70.21%
ROC-AUC : 0.5916 (Ranking Ability)
F1-Score: 0.4167 (Balance Precision/Recall)

Classification Report:
              precision    recall  f1-score   support

           0       0.78      0.82      0.80        34
           1       0.45      0.38      0.42        13

    accuracy                           0.70        47
   macro avg       0.62      0.60      0.61        47
weighted avg       0.69      0.70      0.69        47


Feature Importances:
----------------------------------------
fuel_stress_index              : 0.3556
share_logistics_terms          : 0.2510
price_sum_diesel               : 0.1345
total_messages                 : 0.0680
unique_messages                : 0.0402
count_logistics_terms          : 0.0337
price_sum                      : 0.0238
price_sum_gasoline             : 0.0234
avg_price                      : 0.0222
price_mentions                 : 0.0215
keyword_mentions               : 0.0159
share_diesel_terms             : 0.0057
count_diesel_terms             : 0.0033
share_gasoline_terms           : 0.0007
price_mentions_gasoline        : 0.0005
price_mentions_diesel          : 0.0000
count_gasoline_terms           : 0.0000
count_price_terms              : 0.0000
count_queue_terms              : 0.0000
sentiment_mean                 : 0.0000
share_price_terms              : 0.0000
count_policy_terms             : 0.0000
count_shortage_terms           : 0.0000
share_shortage_terms           : 0.0000
count_agro_terms               : 0.0000
count_rationing_terms          : 0.0000
share_queue_terms              : 0.0000
share_infra_terms              : 0.0000
count_infra_terms              : 0.0000
share_rationing_terms          : 0.0000
share_agro_terms               : 0.0000
sentiment_min                  : 0.0000
share_policy_terms             : 0.0000
----------------------------------------
=== mlp ===
Training Classifier (mlp) on target: crisis_7d
Optimized Decision Threshold: 0.45 (Max Train F1: 0.13)
Accuracy: 48.94%
ROC-AUC : 0.6143 (Ranking Ability)
F1-Score: 0.4545 (Balance Precision/Recall)

Classification Report:
              precision    recall  f1-score   support

           0       0.81      0.38      0.52        34
           1       0.32      0.77      0.45        13

    accuracy                           0.49        47
   macro avg       0.57      0.58      0.49        47
weighted avg       0.68      0.49      0.50        47

Model does not expose feature importance.
=== xgb ===
Training Classifier (xgb) on target: crisis_7d
Running Randomized Search for XGBoost hyperparameters...
Fitting 3 folds for each of 20 candidates, totalling 60 fits
Best Parameters: {'colsample_bytree': np.float64(0.6232334448672797), 'gamma': np.float64(0.4330880728874676), 'learning_rate': np.float64(0.19033450352296263), 'max_depth': 5, 'n_estimators': 199, 'scale_pos_weight': 15, 'subsample': np.float64(0.9879639408647978)}
Optimized Decision Threshold: 0.85 (Max Train F1: 0.75)
Accuracy: 68.09%
ROC-AUC : 0.5509 (Ranking Ability)
F1-Score: 0.3478 (Balance Precision/Recall)

Classification Report:
              precision    recall  f1-score   support

           0       0.76      0.82      0.79        34
           1       0.40      0.31      0.35        13

    accuracy                           0.68        47
   macro avg       0.58      0.57      0.57        47
weighted avg       0.66      0.68      0.67        47


Feature Importances:
----------------------------------------
share_logistics_terms          : 0.2284
price_sum_diesel               : 0.1169
price_mentions_diesel          : 0.1078
total_messages                 : 0.1059
price_sum                      : 0.0875
avg_price                      : 0.0826
count_logistics_terms          : 0.0741
share_diesel_terms             : 0.0561
fuel_stress_index              : 0.0552
keyword_mentions               : 0.0462
price_mentions                 : 0.0158
price_sum_gasoline             : 0.0157
price_mentions_gasoline        : 0.0078
count_shortage_terms           : 0.0000
unique_messages                : 0.0000
count_rationing_terms          : 0.0000
sentiment_min                  : 0.0000
share_rationing_terms          : 0.0000
share_price_terms              : 0.0000
count_agro_terms               : 0.0000
count_price_terms              : 0.0000
count_gasoline_terms           : 0.0000
count_infra_terms              : 0.0000
share_policy_terms             : 0.0000
share_infra_terms              : 0.0000
share_gasoline_terms           : 0.0000
sentiment_mean                 : 0.0000
share_queue_terms              : 0.0000
count_policy_terms             : 0.0000
count_diesel_terms             : 0.0000
count_queue_terms              : 0.0000
share_shortage_terms           : 0.0000
share_agro_terms               : 0.0000
----------------------------------------
