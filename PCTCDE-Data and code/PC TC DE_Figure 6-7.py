#!/usr/bin/env python
# coding: utf-8

# In[87]:


# Select 3 source domain battery quantities with transfer
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

source_excel_path = r"TJU_CY25_05_1_features_all_1.xlsx"
target_excel_path = r"TJU_CY35_05_1_features.xlsx"

feature_cols = ['P1_x', 'P1_y', 'P2_x', 'P2_y', 'P12_Ar'] 
label_col = 'Cap'
n_runs = 10      
n_source = 3     

all_sheets = pd.ExcelFile(source_excel_path).sheet_names

target_df = pd.read_excel(target_excel_path)[10:-100]
X_target = target_df[feature_cols].values
y_target = target_df[label_col].values
n_total = len(X_target)
n_labeled = n_total // 4

X_target_labeled = X_target[:n_labeled]
y_target_labeled = y_target[:n_labeled]
X_target_unlabeled = X_target[n_labeled:]
y_target_unlabeled = y_target[n_labeled:]

def train_and_predict(source_sheets):
  
    source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in source_sheets]
    source_df = pd.concat(source_df_list, ignore_index=True)

    X_source = source_df[feature_cols].values
    y_source = source_df[label_col].values

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_source, X_target_labeled]))
    X_source = scaler.transform(X_source)
    X_target_labeled_scaled = scaler.transform(X_target_labeled)
    X_target_unlabeled_scaled = scaler.transform(X_target_unlabeled)

    dtrain_source = xgb.DMatrix(X_source, label=y_source)
    dtrain_target_labeled = xgb.DMatrix(X_target_labeled_scaled, label=y_target_labeled)
    dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled_scaled)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # Step1: Train in source domain
    model = xgb.train(params, dtrain_source, num_boost_round=300)
    # Step2: Fine-tune
    model = xgb.train(params, dtrain_target_labeled, num_boost_round=100, xgb_model=model)
    # Step3: Estimate
    y_pred = model.predict(dtest_target_unlabeled)
    return y_pred

predictions = []
for i in range(n_runs):
    chosen_sheets = random.sample(all_sheets, n_source)
    print(f"Run {i+1}: chosen {chosen_sheets}")
    y_pred = train_and_predict(chosen_sheets)
    predictions.append(y_pred)

predictions = np.array(predictions)  # (n_runs, n_unlabeled)

mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)

initial_capacity = y_target[0]
true_soh = y_target / initial_capacity
mean_soh = mean_pred / initial_capacity
std_soh = std_pred / initial_capacity

cycles = np.arange(n_labeled, n_total)

plt.figure(figsize=(6,4), dpi=300)
plt.plot(np.arange(len(true_soh)), true_soh, color='black', label='True SOH')
plt.plot(cycles, mean_soh, color='blue', label='Predicted SOH (Mean of 10 runs)')
plt.fill_between(cycles, mean_soh-std_soh, mean_soh+std_soh, color='blue', alpha=0.2, label='±1 std')
plt.xlabel('Cycle')
plt.ylabel('SOH (Normalized Capacity)')
plt.title('SOH Prediction with Random Source Selection (10 runs)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[82]:


#TJU no transfer
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

source_sheets = ["Battery_#1"]
#source_sheets = ["Battery_#1", "Battery_#2"]
source_sheets = ["Battery_#1", "Battery_#2","Battery_#3"]
source_excel_path = "TJU_CY25_05_1_features_all_1.xlsx"
source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in source_sheets]
source_df = pd.concat(source_df_list, ignore_index=True)

target_df = pd.read_excel(r"TJU_CY35_05_1_features.xlsx")[10:-100]

feature_cols = ['P1_x','P1_y','P2_x', 
                'P2_y', 'P12_Ar']
label_col = 'Cap'

X_source = source_df[feature_cols].values
y_source = source_df[label_col].values

split_index = int(len(target_df) * 1/4)  
target_df_subset = target_df.iloc[split_index:]  

X_target = target_df_subset[feature_cols].values
y_target = target_df_subset[label_col].values

scaler = StandardScaler()

scaler.fit(np.vstack([X_source, target_df[feature_cols].values]))
X_source = scaler.transform(X_source)
X_target = scaler.transform(X_target)

dtrain_source = xgb.DMatrix(X_source, label=y_source)
dtest_target = xgb.DMatrix(X_target)  

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

print("🚀 Training model on source domain only...")
model = xgb.train(params, dtrain_source, num_boost_round=300)

print("📊 Predicting on 3/4 target domain...")
y_pred = model.predict(dtest_target)

output_df = pd.DataFrame({
    "Cycle": target_df_subset["cycle_number"].values,
    "Predicted_Capacity": y_pred,
    "True_Capacity": y_target
})

output_df.to_excel(r"TJU_xgb_predicted_target_no_transfer.xlsx", index=False)

if len(y_pred) == len(y_target):
    mse = mean_squared_error(y_target, y_pred)
    mae = mean_absolute_error(y_target, y_pred)
    print(f"Evaluation Matrics：\nMSE = {mse:.4f}, MAE = {mae:.4f}")
    print(f"Estimated Samples: {len(y_pred)}")


# In[89]:


#Select the number of batteries in three source domains no transfer 
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

source_excel_path = r"TJU_CY25_05_1_features_all_1.xlsx"
target_excel_path = r"TJU_CY35_05_1_features.xlsx"

feature_cols = ['P1_x', 'P1_y', 'P2_x', 'P2_y', 'P12_Ar']
label_col = 'Cap'
n_runs = 10      
n_source = 3     

all_sheets = pd.ExcelFile(source_excel_path).sheet_names

target_df = pd.read_excel(target_excel_path)[10:-100]
X_target_all = target_df[feature_cols].values
y_target_all = target_df[label_col].values
n_total = len(X_target_all)
n_labeled = n_total // 4  

X_target_unlabeled = X_target_all[n_labeled:]
y_target_unlabeled = y_target_all[n_labeled:]

def train_and_predict(source_sheets):
  
    source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in source_sheets]
    source_df = pd.concat(source_df_list, ignore_index=True)
    X_source = source_df[feature_cols].values
    y_source = source_df[label_col].values

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_source, X_target_all]))
    X_source_scaled = scaler.transform(X_source)
    X_target_unlabeled_scaled = scaler.transform(X_target_unlabeled)

    dtrain_source = xgb.DMatrix(X_source_scaled, label=y_source)
    dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled_scaled)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    model = xgb.train(params, dtrain_source, num_boost_round=300)

    y_pred = model.predict(dtest_target_unlabeled)
    return y_pred

predictions = []
for i in range(n_runs):
    chosen_sheets = random.sample(all_sheets, n_source)
    print(f"Run {i+1}: chosen {chosen_sheets}")
    y_pred = train_and_predict(chosen_sheets)
    predictions.append(y_pred)

predictions = np.array(predictions)  

mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)

initial_capacity = y_target_all[0]
true_soh = y_target_all / initial_capacity
mean_soh = mean_pred / initial_capacity
std_soh = std_pred / initial_capacity

cycles = np.arange(n_labeled, n_total)

plt.figure(figsize=(6,4), dpi=300)
plt.plot(np.arange(len(true_soh)), true_soh, color='black', label='True SOH')
plt.plot(cycles, mean_soh, color='red', label='Predicted SOH (Mean of 10 runs)')
plt.fill_between(cycles, mean_soh-std_soh, mean_soh+std_soh, color='red', alpha=0.2, label='±1 std')
plt.xlabel('Cycle')
plt.ylabel('SOH (Normalized Capacity)')
plt.title('SOH Prediction (No Transfer, 10 Random Runs)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[72]:


#TJU XGboost with transfer P1 features
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

source_sheets = ["Battery_#1"]
#source_sheets = ["Battery_#1", "Battery_#2"]
source_sheets = ["Battery_#1", "Battery_#2", "Battery_#3"]
source_excel_path = "TJU_CY25_05_1_features_all_1.xlsx"
source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in source_sheets]
source_df = pd.concat(source_df_list, ignore_index=True)

target_df = pd.read_excel(r"TJU_CY35_05_1_features.xlsx")[10:-100]
feature_cols = ['P1_x','P1_y']
#feature_cols = ['Cycle','First_Peak_dQdV','Voltage_at_First','Second_Peak_dQdV', 'Voltage_at_Second', 'Area_3.7_4.2']
#feature_cols = ['First_Peak_dQdV']
label_col = 'Cap'

X_source = source_df[feature_cols].values
y_source = source_df[label_col].values

X_target = target_df[feature_cols].values
y_target = target_df[label_col].values

n_total = len(X_target)
n_labeled = n_total // 4

X_target_labeled = X_target[:n_labeled]
y_target_labeled = y_target[:n_labeled]

X_target_unlabeled = X_target[n_labeled:]
y_target_unlabeled = y_target[n_labeled:]  

scaler = StandardScaler()
scaler.fit(np.vstack([X_source, X_target_labeled]))
X_source = scaler.transform(X_source)
X_target_labeled = scaler.transform(X_target_labeled)
X_target_unlabeled = scaler.transform(X_target_unlabeled)

dtrain_source = xgb.DMatrix(X_source, label=y_source)
dtrain_target_labeled = xgb.DMatrix(X_target_labeled, label=y_target_labeled)
dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

print("🔧 Training on source domain...")
model = xgb.train(params, dtrain_source, num_boost_round=300)

print("🔧 Fine-tuning on target domain (1/4)...")
model = xgb.train(params, dtrain_target_labeled, num_boost_round=100, xgb_model=model)

y_pred = model.predict(dtest_target_unlabeled)

output_df = pd.DataFrame({
    "Cycle": np.arange(n_labeled, n_total),
    "Predicted_Capacity": y_pred,
    "True_Capacity": y_target[n_labeled:]  
})
output_df.to_excel(r"TJU_xgb_predicted_target_P1_with_transfer.xlsx", index=False)


# In[36]:


#Number of 3-source domain batteries with transfer Use peak1 features 
import pandas as pd
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

source_excel_path = r"TJU_CY25_05_1_features_all_1.xlsx"
target_excel_path = r"TJU_CY35_05_1_features.xlsx"

feature_cols = ['P1_x', 'P1_y']
label_col = 'Cap'
n_runs = 10     
n_source = 3     

all_sheets = pd.ExcelFile(source_excel_path).sheet_names

target_df = pd.read_excel(target_excel_path)[10:-100]
X_target = target_df[feature_cols].values
y_target = target_df[label_col].values
n_total = len(X_target)
n_labeled = n_total // 4

X_target_labeled = X_target[:n_labeled]
y_target_labeled = y_target[:n_labeled]
X_target_unlabeled = X_target[n_labeled:]
y_target_unlabeled = y_target[n_labeled:]

def train_and_predict(source_sheets):
   
    source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in source_sheets]
    source_df = pd.concat(source_df_list, ignore_index=True)

    X_source = source_df[feature_cols].values
    y_source = source_df[label_col].values

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_source, X_target_labeled]))
    X_source = scaler.transform(X_source)
    X_target_labeled_scaled = scaler.transform(X_target_labeled)
    X_target_unlabeled_scaled = scaler.transform(X_target_unlabeled)

    dtrain_source = xgb.DMatrix(X_source, label=y_source)
    dtrain_target_labeled = xgb.DMatrix(X_target_labeled_scaled, label=y_target_labeled)
    dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled_scaled)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # Step1: Train in source domain
    model = xgb.train(params, dtrain_source, num_boost_round=300)
    # Step2: Fine-tune
    model = xgb.train(params, dtrain_target_labeled, num_boost_round=100, xgb_model=model)
    # Step3: Estimate
    y_pred = model.predict(dtest_target_unlabeled)
    return y_pred

predictions = []
for i in range(n_runs):
    chosen_sheets = random.sample(all_sheets, n_source)
    print(f"Run {i+1}: chosen {chosen_sheets}")
    y_pred = train_and_predict(chosen_sheets)
    predictions.append(y_pred)

predictions = np.array(predictions)  

mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)

initial_capacity = y_target[0]
true_soh = y_target / initial_capacity
mean_soh = mean_pred / initial_capacity
std_soh = std_pred / initial_capacity

cycles = np.arange(n_labeled, n_total)

plt.figure(figsize=(6,4), dpi=300)

plt.plot(np.arange(len(true_soh)), true_soh, color='black', label='True SOH')

plt.plot(cycles, mean_soh, color='blue', label='Predicted SOH (Mean of 10 runs)')
plt.fill_between(cycles, mean_soh-std_soh, mean_soh+std_soh, color='blue', alpha=0.2, label='±1 std')

plt.xlabel('Cycle')
plt.ylabel('SOH (Normalized Capacity)')
plt.title('SOH Prediction with Random Source Selection (10 runs)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[110]:


# Plot Figure in three conditions
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

source_excel_path = r"TJU_CY25_05_1_features_all_1.xlsx"
target_excel_path = r"TJU_CY35_05_1_features.xlsx"
target_df = pd.read_excel(target_excel_path)[10:-100]

feature_cols_1 = ['P1_x', 'P1_y', 'P2_x', 'P2_y', 'P12_Ar']
feature_cols_2 = ['P1_x', 'P1_y', 'P2_x', 'P2_y', 'P12_Ar']
feature_cols_3 = ['P1_x', 'P1_y']
label_col = 'Cap'
n_runs = 10
n_source = 3

all_sheets = pd.ExcelFile(source_excel_path).sheet_names

random.seed(55)
chosen_sets = [random.sample(all_sheets, n_source) for _ in range(n_runs)]
print("Fixed source domain combination：", chosen_sets)

X_target_all_1 = target_df[feature_cols_1].values
X_target_all_2 = target_df[feature_cols_2].values
X_target_all_3 = target_df[feature_cols_3].values
y_target_all = target_df[label_col].values

n_total = len(y_target_all)
n_labeled = n_total // 4

X_target_labeled_1 = X_target_all_1[:n_labeled]
X_target_unlabeled_1 = X_target_all_1[n_labeled:]

X_target_labeled_2 = X_target_all_2[:n_labeled]
X_target_unlabeled_2 = X_target_all_2[n_labeled:]

X_target_labeled_3 = X_target_all_3[:n_labeled]
X_target_unlabeled_3 = X_target_all_3[n_labeled:]

y_target_labeled = y_target_all[:n_labeled]
y_target_unlabeled = y_target_all[n_labeled:]

initial_capacity = y_target_all[0]
true_soh = y_target_all / initial_capacity
cycles = np.arange(n_labeled, n_total)

def predict_with_transfer(source_sheets):
    """with transfer（feature_cols_1）"""
    source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in source_sheets]
    source_df = pd.concat(source_df_list, ignore_index=True)
    X_source = source_df[feature_cols_1].values
    y_source = source_df[label_col].values

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_source, X_target_labeled_1]))
    X_source_scaled = scaler.transform(X_source)
    X_target_labeled_scaled = scaler.transform(X_target_labeled_1)
    X_target_unlabeled_scaled = scaler.transform(X_target_unlabeled_1)

    dtrain_source = xgb.DMatrix(X_source_scaled, label=y_source)
    dtrain_target_labeled = xgb.DMatrix(X_target_labeled_scaled, label=y_target_labeled)
    dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled_scaled)

    params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42}

    model = xgb.train(params, dtrain_source, num_boost_round=300)
    model = xgb.train(params, dtrain_target_labeled, num_boost_round=100, xgb_model=model)
    return model.predict(dtest_target_unlabeled)

def predict_no_transfer(source_sheets):
    """no transfer（feature_cols_2）"""
    source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in source_sheets]
    source_df = pd.concat(source_df_list, ignore_index=True)
    X_source = source_df[feature_cols_2].values
    y_source = source_df[label_col].values

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_source, X_target_all_2]))
    X_source_scaled = scaler.transform(X_source)
    X_target_unlabeled_scaled = scaler.transform(X_target_unlabeled_2)

    dtrain_source = xgb.DMatrix(X_source_scaled, label=y_source)
    dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled_scaled)

    params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42}

    model = xgb.train(params, dtrain_source, num_boost_round=300)
    return model.predict(dtest_target_unlabeled)

def predict_F12_transfer(source_sheets):
    """with transfer + P1 features（feature_cols_3）"""
    source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in source_sheets]
    source_df = pd.concat(source_df_list, ignore_index=True)
    X_source = source_df[feature_cols_3].values
    y_source = source_df[label_col].values

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_source, X_target_labeled_3]))
    X_source_scaled = scaler.transform(X_source)
    X_target_labeled_scaled = scaler.transform(X_target_labeled_3)
    X_target_unlabeled_scaled = scaler.transform(X_target_unlabeled_3)

    dtrain_source = xgb.DMatrix(X_source_scaled, label=y_source)
    dtrain_target_labeled = xgb.DMatrix(X_target_labeled_scaled, label=y_target_labeled)
    dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled_scaled)

    params = {'objective': 'reg:squarederror', 'max_depth': 6, 'eta': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 42}

    model = xgb.train(params, dtrain_source, num_boost_round=300)
    model = xgb.train(params, dtrain_target_labeled, num_boost_round=100, xgb_model=model)
    return model.predict(dtest_target_unlabeled)

def run_predictions(predict_func):
    preds = []
    for sheets in chosen_sets:
        preds.append(predict_func(sheets))
    preds = np.array(preds)
    return np.mean(preds, axis=0), np.std(preds, axis=0)

mean_transfer, std_transfer = run_predictions(predict_with_transfer)
mean_no_transfer, std_no_transfer = run_predictions(predict_no_transfer)
mean_F12, std_F12 = run_predictions(predict_F12_transfer)

mean_transfer /= initial_capacity
std_transfer /= initial_capacity
mean_no_transfer /= initial_capacity
std_no_transfer /= initial_capacity
mean_F12 /= initial_capacity
std_F12 /= initial_capacity

plt.figure(figsize=(7,5), dpi=300)
plt.plot(np.arange(len(true_soh)), true_soh, color='black', label='True SOH')

plt.plot(cycles, mean_transfer, color='blue', label='With Transfer')
plt.fill_between(cycles, mean_transfer-std_transfer, mean_transfer+std_transfer,
                 color='blue', alpha=0.2)

plt.plot(cycles, mean_no_transfer, color='red', label='No Transfer')
plt.fill_between(cycles, mean_no_transfer-std_no_transfer, mean_no_transfer+std_no_transfer,
                 color='red', alpha=0.2)

plt.plot(cycles, mean_F12, color='green', label='F12 Transfer')
plt.fill_between(cycles, mean_F12-std_F12, mean_F12+std_F12,
                 color='green', alpha=0.2)

plt.xlabel('Cycle')
plt.ylabel('SOH (Normalized Capacity)')
plt.title('SOH Prediction Comparison (10 Random Runs)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[111]:


plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = True  
plt.figure(figsize=(3,2), dpi=300)
plt.plot(np.arange(len(true_soh)), true_soh, color='gray', label='True SOH')

plt.plot(cycles, mean_transfer, color='#a79fce', label='With Transfer')
plt.fill_between(cycles, mean_transfer-std_transfer, mean_transfer+std_transfer,
                 color='#a79fce', alpha=0.2)

plt.plot(cycles, mean_no_transfer, color='#8ebcdb', label='No Transfer')
plt.fill_between(cycles, mean_no_transfer-std_no_transfer, mean_no_transfer+std_no_transfer,
                 color='#8ebcdb', alpha=0.2)

plt.plot(cycles, mean_F12, color='#fe9f69', label='F12 Transfer')
plt.fill_between(cycles, mean_F12-std_F12, mean_F12+std_F12,
                 color='#fe9f69', alpha=0.2)

plt.xlabel('Lifetime (Cycles)', fontsize=11)
plt.ylabel('SOH (a.u.)', fontsize=11)
plt.title('SOH Estimation Using One Battery in TJU')

#plt.legend()
#plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[113]:


#Plot Errors
import matplotlib.pyplot as plt
import numpy as np

error_transfer = true_soh[n_labeled:] - mean_transfer
error_no_transfer = true_soh[n_labeled:] - mean_no_transfer
error_F12 = true_soh[n_labeled:] - mean_F12

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = True

plt.figure(figsize=(3,2), dpi=300)

plt.plot(cycles, error_transfer, color='#a79fce', label='With Transfer Error')
plt.fill_between(cycles,
                 error_transfer-std_transfer,
                 error_transfer+std_transfer,
                 color='#a79fce', alpha=0.2)

plt.plot(cycles, error_no_transfer, color='#8ebcdb', label='No Transfer Error')
plt.fill_between(cycles,
                 error_no_transfer-std_no_transfer,
                 error_no_transfer+std_no_transfer,
                 color='#8ebcdb', alpha=0.2)

plt.plot(cycles, error_F12, color='#fe9f69', label='F12 Transfer Error')
plt.fill_between(cycles,
                 error_F12-std_F12,
                 error_F12+std_F12,
                 color='#fe9f69', alpha=0.2)

#plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

plt.xlabel('Lifetime (Cycles)', fontsize=11)
plt.ylabel('Absolute Error (a.u.)', fontsize=11)
plt.title('Estimation Error Using Three Batteries')
plt.ylim(-0.09,0.05)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

true_soh_pred = true_soh[n_labeled:]

def calc_metrics(pred, name):
    mae = mean_absolute_error(true_soh_pred, pred)
    rmse = mean_squared_error(true_soh_pred, pred, squared=False)
    mape = np.mean(np.abs((true_soh_pred - pred) / true_soh_pred)) * 100
    print(f"{name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

calc_metrics(mean_transfer, "With Transfer")
calc_metrics(mean_no_transfer, "No Transfer")
calc_metrics(mean_F12, "F12 Transfer")


# In[61]:


# DE Analysis across source domain battery numbers and feature numbers
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random

source_excel_path = r"TJU_CY25_05_1_features_all_1.xlsx"
target_df = pd.read_excel(r"TJU_CY35_05_1_features.xlsx")[10:-100]
initial_capacity = target_df['Cap'].iloc[0] 

exclude_sheets = ["Battery_#4","Battery_#5","Battery_#8","Battery_#9","Battery_#13"]
excel_file = pd.ExcelFile(source_excel_path)
valid_sheets = [s for s in excel_file.sheet_names if s not in exclude_sheets]

feature_addition_order = [
    ['cycle_number'],
    ['cycle_number', 'P1_x'],
    ['cycle_number', 'P1_x', 'P1_y'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x', 'P2_y'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x', 'P2_y', 'P12_Ar']
]

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

label_col = 'Cap'
n_runs = 10  
random.seed(55)

chosen_sets_dict = {}
for n_batteries in range(1, len(valid_sheets) + 1):
    chosen_sets_dict[n_batteries] = [random.sample(valid_sheets, n_batteries) for _ in range(n_runs)]
print("Fixed random combinations have been generated")

# ==== 结果矩阵 ====
heatmap_data = np.zeros((len(valid_sheets), len(feature_addition_order)))

for n_batteries in range(1, len(valid_sheets) + 1):
    for j, feature_cols in enumerate(feature_addition_order):
        run_rmse = []
        for selected_sheets in chosen_sets_dict[n_batteries]:
           
            source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in selected_sheets]
            combined_source_df = pd.concat(source_df_list, ignore_index=True)

            X_source = combined_source_df[feature_cols].values
            y_source = combined_source_df[label_col].values
            X_target = target_df[feature_cols].values
            y_target = target_df[label_col].values

            scaler = StandardScaler()
            scaler.fit(np.vstack([X_source, X_target]))
            X_source = scaler.transform(X_source)
            X_target = scaler.transform(X_target)

            dtrain = xgb.DMatrix(X_source, label=y_source)
            dtest = xgb.DMatrix(X_target)
            model = xgb.train(params, dtrain, num_boost_round=300)
            y_pred = model.predict(dtest)

            soh_true = y_target / initial_capacity
            soh_pred = y_pred / initial_capacity
            rmse_soh = np.sqrt(np.mean((soh_true - soh_pred) ** 2))

            run_rmse.append(rmse_soh)

        heatmap_data[n_batteries-1, j] = np.mean(run_rmse)

plt.figure(figsize=(6, 5), dpi=600)
sns.heatmap(heatmap_data, annot=True, fmt=".2e", cmap="YlGnBu", cbar_kws={'label': 'RMSE (SOH)'})

plt.xlabel("Number of Features Added", fontsize=14)
plt.ylabel("Number of Source Batteries Used", fontsize=14)
plt.title(f"SOH RMSE Heatmap (Average over {n_runs} Fixed Random Runs)", fontsize=16)

plt.xticks(ticks=np.arange(len(feature_addition_order)) + 0.5,
           labels=[len(f) for f in feature_addition_order],
           rotation=0, fontsize=12)
plt.yticks(ticks=np.arange(len(valid_sheets)) + 0.5,
           labels=np.arange(1, len(valid_sheets) + 1),
           rotation=0, fontsize=12)

plt.tight_layout()
plt.show()


# In[82]:


plt.figure(figsize=(3, 2), dpi=600)
sns.heatmap(
    heatmap_data,
    annot=False,
    fmt=".2e",
    cmap="YlGnBu",
    cbar_kws={'label': 'RMSE of SOH'},
    square=False,
    linewidths=0.2,
    linecolor='white'
)

plt.xlabel("Number of Features Added", fontsize=10)
plt.ylabel("Number of Source Batteries", fontsize=10)
plt.title("SOH RMSE Heatmap", fontsize=11)

plt.xticks(
    ticks=np.arange(len(feature_addition_order)) + 0.5,
    labels=[len(f) for f in feature_addition_order],
    rotation=0,
    fontsize=9
)

yticks_pos = np.arange(len(valid_sheets)) + 0.5
yticks_pos_odd = [pos for i, pos in enumerate(yticks_pos, start=1) if i % 2 == 1]
yticks_label_odd = [i for i in range(1, len(valid_sheets) + 1) if i % 2 == 1]

plt.yticks(
    ticks=yticks_pos_odd,
    labels=yticks_label_odd,
    rotation=0,
    fontsize=10
)

cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=8)       
cbar.set_label('RMSE of SOH', fontsize=8)  

plt.tight_layout(pad=0.5)
plt.show()


# In[14]:


#PC TC DE Single feature
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

features = {
    "P1_x": [4, 1, 5],
    "P1_y": [3, 4, 5],
    "P2_x": [2, 3, 3],
    "P2_y": [1, 5, 3],
    "P12_Ar": [5, 2, 1]
}

colors = ["#A7B9D7", "#FCDCB4", "#DEA3A2", "#508AC7", "#D3E3F2"]
dimensions = ["PC", "TC", "DE"]
angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(1, 5, figsize=(7, 2), subplot_kw=dict(polar=True), dpi=600)

for ax, (feature, values), color in zip(axes, features.items(), colors):
    data = values + values[:1]
    ax.plot(angles, data, color=color, linewidth=2)
    ax.fill(angles, data, color=color, alpha=0.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=13)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8)
    ax.set_ylim(0, 5)
    ax.set_title(feature, fontsize=14, pad=10)

plt.tight_layout()
plt.show()


# In[42]:


#PC TC DE Feature combinations
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

features = {
    "P1_x, P1_y": [4, 4, 5],
    "P1_x, P1_y, P2_x": [4, 4, 3],
    "P1_x, P1_y, P2_x, P2_y": [4, 5, 3],
    "P1_x, P1_y, P2_x, P2_y, P12_Ar": [5, 5, 1]
}

features = {
    "Combination 1": [4, 4, 5],
    "Combination 2": [4, 4, 3],
    "Combination 3": [4, 5, 3],
    "Combination 4": [5, 5, 1]
}

dimensions = ["PC", "TC", "DE"]
angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
angles += angles[:1] 

colors = ["#88CEE6", "#B2D3A4", "#B696B6", "#80C1C4"]

# 绘制单独的雷达图
fig, axes = plt.subplots(1, 4, figsize=(5.6, 2), dpi=600, subplot_kw=dict(polar=True))
for ax, (feature, values), color in zip(axes, features.items(), colors):
    data = values + values[:1]
    ax.plot(angles, data, color=color, linewidth=2)
    ax.fill(angles, data, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=11)  # 与五张图一致
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=7)
    ax.set_ylim(0, 5)
    ax.set_title(feature, fontsize=12, pad=10)
plt.tight_layout()
plt.show()

