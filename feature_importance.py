# -*- coding: utf-8 -*-
import random
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Input, Subtract, Add
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import losses
import keras.backend as K
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
from scipy import stats
from lib.treatments import hypertension_treatments, dm_treatment, ascvd_treatment, hypertension_treatments_history, dm_treatment_history, framingham_ascvd_risk
from lib.DQN import DQNAgent
from lib.next_state_cols import next_state_cols

target_treatment = None
action_size = 100

excluded = ['study_id', 'encounter_dt_ran']

data_path = '~/Research/PHD/project/Hua Zheng/previous code/cleaned_EHR_treatment_param_lab_test_final_3diseases-v6.csv'
# '~/Research/PHD/project/Hua Zheng/previous code/cleaned_EHR_treatment_param_lab_test_final_3diseases_cvd_encounter_diagnosis.csv'
data = pd.read_csv(data_path)

data = data.drop(['egfr_mdrd_african_american_min', 'egfr_mdrd_african_american_max', 'egfr_mdrd_african_american',
                  'egfr_mdrd_non_african_american', 'egfr_mdrd_non_african_american_max',
                  'egfr_mdrd_non_african_american_min', 'next_egfr_mdrd_african_american',
                  'next_egfr_mdrd_african_american_max', 'next_egfr_mdrd_african_american_min',
                  'next_egfr_mdrd_non_african_american', 'next_egfr_mdrd_non_african_american_max',
                  'next_egfr_mdrd_non_african_american_min'], axis=1)
data = data.dropna()

if diagnosis_reward:
    label_cols = list(data.columns[116:159])
else:
    label_cols = list(data.columns[117:160])

full_label_cols = label_cols
if target_treatment != None:
    label_cols = treatment_ctg[target_treatment]
    action_size = len(label_cols)

#data = data.loc[data[label_cols].sum(axis=1, skipna=True) != 0,]
data = data.loc[(data[label_cols].sum(axis=1, skipna=True) != 0) & (
            data[set(full_label_cols) - set(label_cols)].sum(axis=1, skipna=True) == 0),]

target = data[label_cols].apply(lambda x: hash_to_action(x), axis=1)
data['target'] = target
counter = collections.Counter(target)
target_set = list([i[0] for i in counter.most_common(action_size)])
target_replacement = dict(zip(iter(target_set), range(action_size)))
target_column_renames = ['target' + str(i) for i in range(action_size)]

data = data[data.target.apply(lambda x: x in target_set)]
data['target'] = data['target'].replace(target_replacement)
reward_cols = ['reward']


# %%
state_cols = [s[5:] for s in next_state_cols]
# state_cols = list(set(data.columns) - set(full_label_cols) - {'target'} - set(
#     ['reward', 'reward_bp', 'reward_ascvd', 'reward_diabetes', 'risk_ascvd', 'next_risk_ascvd']) - set(
#     ['study_id', 'encounter_dt_ran']) - set(next_state_cols))
if diagnosis_reward:
    state_cols = state_cols - (['CVD', 'days_to_CVD'])

# next_state_cols = ['next_' + s for s in list(state_cols)]
patients_column = data[['study_id', 'encounter_dt_ran']]
#data = data.drop(excluded, axis=1)
_temp = data.drop('target', axis=1).max(skipna=True) - data.drop('target', axis=1).min(skipna=True)
_temp[data.drop('target', axis=1).columns[(_temp == 0).values]] = 1.0
normalized_df = (data.drop(['target'], axis=1) - data.drop('target', axis=1).min(skipna=True)) / _temp

# constant_variable_df = data.drop('target', axis=1).max(skipna=True) - data.drop('target', axis=1).min(skipna=True)
# dropped_state = normalized_df.columns[(constant_variable_df == 0).values]
# normalized_df[dropped_state] = data[dropped_state]
normalized_df['reward'] = (data['reward'] - data['reward'].min(skipna=True)) / (data['reward'].max() / 3)
# next_state_cols = set(next_state_cols) - set(dropped_state)
# state_cols = set(state_cols) - set(dropped_state)
ohe = to_categorical(data['target'], action_size)

for i, col in enumerate(target_column_renames):
    normalized_df[col] = ohe[:, i]
normalized_df[['study_id', 'encounter_dt_ran']] = patients_column
train, test = train_test_split(normalized_df, test_size=0.4, random_state=2019)
train2, test2 = train_test_split(data, test_size=0.4, random_state=2019)

# x_train = train[state_cols]
# y_train_class = train[label_cols]
# y_train_category = train[['RX_CATEGORY']]

# x_test = test[state_cols]
# y_test_class = test[label_cols]

state_size = len(state_cols)
action_size = len(target_column_renames)

agent = DQNAgent(state_size, action_size, target_column_renames, state_cols, reward_cols, next_state_cols)
agent.model = agent.save("./3d256-512-256-episodes20000--dqn-mse-target_treatmentNone")

from IPython.display import display

# agent.model = load_model('model-5rounds-l3-tanh-lr0004.h5')
x_train = train[state_cols]
x_test = test[state_cols]

''' obtain AI feature importance
'''
background = x_train.values[np.random.choice(x_train.shape[0], 100, replace=False)]
e = shap.DeepExplainer(agent.model, background)
shap_values = e.shap_values(x_test.values[:100])

# takes long time thus save first
np.save('shap_values',shap_values)

''' obtain clinicians' feature importance
'''
model = xgb.XGBClassifier(max_depth=2,
                              n_estimators=300,
                              num_class= 100,
                              colsample_bytree=0.1,
                              subsample=0.8,
                              learning_rate=0.1,
                              objective='multi:softmax',
                              gpu_id=0,
                              eval_metric=['mlogloss'],
                              n_jobs=4)
model.fit(np.array(train[state_cols]), np.array(train2['target']),
              eval_set=[(np.array(test[state_cols]), np.array(test2['target']))],
              verbose=1,
              early_stopping_rounds=4)
mybooster = model.get_booster()

model_bytearray = mybooster.save_raw()[4:]
def myfun(self=None):
    return model_bytearray
mybooster.save_raw = myfun

# Shap explainer initilization

shap_ex = shap.TreeExplainer(mybooster)
e = shap_ex.shap_values(np.array(test[state_cols])[np.random.choice(test.shape[0], 5000, replace=False)])

clinician_shap_values = pd.DataFrame(np.abs(np.array(e)).mean(0))
clinician_shap_values = np.abs(clinician_shap_values)
k_clinician=pd.DataFrame(clinician_shap_values.mean()).reset_index()


AI_shap_values = np.load('/Users/hua.zheng/PycharmProjects/ReinforcementLearning/results/shap_values.npy')
AI_shap_values = pd.DataFrame(np.abs(np.array(AI_shap_values)).mean(0))
AI_shap_values = np.abs(AI_shap_values)
k=pd.DataFrame(AI_shap_values.mean()).reset_index()

data = pd.DataFrame({'feature':state_cols,'AI': k[0],'clincian': k_clinician[0]})
data = pd.read_csv('raw_feature_importance.csv',index=None)
data.drop('Unnamed: 0',inplace=True,axis=1)
biomarker_impact = data.loc[(~data['feature'].str.contains('hist')) & (~data['feature'].str.contains('cur')),]
treatment_hist = data.loc[(data['feature'].str.contains('hist')) | (data['feature'].str.contains('cur')),]


cleaned_biomarker_impact = biomarker_impact.loc[(~biomarker_impact['feature'].str.contains('max')) & (~biomarker_impact['feature'].str.contains('min')),]

feature_list_ = []
biomarker_list_AI = []
biomarker_list_clincian = []

for bio in cleaned_biomarker_impact['feature'].values:
    if bio == 'hdl_cholesterol':
        feature_list_.append('HDL')
    elif bio == 'ldl_cholesterol':
        feature_list_.append('LDL')
    elif bio == 'cholesterol,_total':
        feature_list_.append('TC')
    elif bio == 'bp_systolic':
        feature_list_.append('Systolic BP')
    elif bio == 'bp_diastolic':
        feature_list_.append('Diastolic BP')
    elif bio == 'time_last_vist':
        feature_list_.append('Time since last vist')
    elif bio == 'bmi':
        feature_list_.append('BMI')
    elif bio == 'hemoglobin_a1c':
        feature_list_.append('HbA1c')
    elif bio == 'sex_male':
        feature_list_.append('Sex(male)')
    elif bio == 'creatinine':
        feature_list_.append('Creatinine')
    elif bio == 'triglycerides':
        feature_list_.append('Triglycerides')
    elif bio == 'age':
        feature_list_.append('Age')
    elif bio == 'smoke':
        feature_list_.append('Smoke')
    elif bio == 'time_to_first_visit':
        feature_list_.append('Time since first visit')
    else:
        feature_list_.append(bio.replace('_', ' '))
    biomarker_list_AI.append(biomarker_impact.loc[biomarker_impact['feature'].str.contains(bio),].mean()['AI'])
    biomarker_list_clincian.append(biomarker_impact.loc[biomarker_impact['feature'].str.contains(bio),].mean()['clincian'])

feature_list_.append('race')
biomarker_list_AI.append(biomarker_impact.loc[biomarker_impact['feature'].str.contains('race'),].mean()['AI'])
biomarker_list_clincian.append(biomarker_impact.loc[biomarker_impact['feature'].str.contains('race'),].mean()['clincian'])
feature_list_.append('Recent Therapies')
biomarker_list_AI.append(treatment_hist['AI'].mean() / 1.5)
biomarker_list_clincian.append(treatment_hist['clincian'].mean())

feature_importance = pd.DataFrame({'variables': feature_list_, 'AI': biomarker_list_AI, 'clinician': biomarker_list_clincian})
feature_importance.iloc[16,1] = feature_importance.iloc[16,]['AI'] / 1.5
feature_importance = feature_importance.loc[(~feature_importance['variables'].str.contains('race')) & (~feature_importance['variables'].str.contains('race')),]

# feature_importance.to_csv('feature_importance3.csv')
# data.to_csv('raw_feature_importance.csv')
#
# import seaborn as sns
# sns.set(style="whitegrid")

feature_group = ['biomarker', 'biomarker', 'biomarker','biomarker','biomarker',
                                       'treatment history', 'biomarker', 'biomarker', 'demographics',
                                       'demographics','biomarker','demographics','biomarker','treatment history','treatment history']

feature_importance['feature_group'] =feature_group
feature_importance.sort_values(by='feature_group',inplace=True)
# Make the PairGrid
feature_importance_temp = feature_importance.copy()
feature_importance_temp['AI'] = feature_importance_temp['AI'].values * 100

g = sns.PairGrid(feature_importance_temp,
                 x_vars=feature_importance.columns[1:3], y_vars=["variables"],
                 height=5, aspect=1)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      # palette="ch:s=1,r=-.1,h=1_r",
      linewidth=1,
      edgecolor="w")

# Use the same x axis limits on all columns and add better labels
# g.set(xlim={(0, 0.02), xlabel={"Crashes", '1'}, ylabel="")



# for ax, title in zip(g.axes.flat, titles):
#
#     # Set a different title for each axes
#     ax.set(title=title)
#
#     # Make the grid horizontal instead of vertical
#     ax.xaxis.grid(False)
#     ax.yaxis.grid(True)


sns.despine(left=True, bottom=True)
plt.savefig('feature_importance.svg')
plt.show()



temp_feature_imp = feature_importance.groupby('feature_group').apply(lambda x: x.sort_values(by='AI',ascending=True))
temp_feature_imp.reset_index()

temp_feature_imp['AI'] = temp_feature_imp['AI'] * temp_feature_imp['clinician'].sum() / temp_feature_imp['AI'].sum()
temp_feature_imp.plot(kind= 'barh' , rot=0, x='variables', figsize=(20,10), fontsize=20)

cmap = plt.get_cmap("tab20c")
outer_colors = cmap(6)
in_colors = cmap(0)

feature_importance.sort_values('AI',ascending=True)[['variables','AI']].plot(kind= 'barh' , rot=0, x='variables', figsize=(12,12), fontsize=20,color=in_colors, alpha=0.9,zorder=2)
plt.show()
# width = 0.4

# temp_feature_imp.AI.plot(kind='barh', color='red', ax=ax, width=width, position=0,x='variables')
# temp_feature_imp.clinician.plot(kind='barh', color='blue', ax=ax2, width=width, position=1,x='variables')
#

# ax2 = ax.twinx()
# seaxs = ax.secondary_xaxis('top', functions=rho2m)
plt.savefig('feature_imp1-ai.png')
plt.savefig('feature_imp2-ai.pdf')
plt.savefig('feature_imp3-ai.svg')

feature_importance.sort_values('clinician',ascending=True)[['variables','clinician']].plot(kind= 'barh' , rot=0, x='variables', figsize=(12,12), fontsize=20,color=outer_colors, alpha=0.9, zorder=2)
plt.show()
# width = 0.4

# temp_feature_imp.AI.plot(kind='barh', color='red', ax=ax, width=width, position=0,x='variables')
# temp_feature_imp.clinician.plot(kind='barh', color='blue', ax=ax2, width=width, position=1,x='variables')
#

# ax2 = ax.twinx()
# seaxs = ax.secondary_xaxis('top', functions=rho2m)
plt.savefig('feature_imp1-ai.png')
plt.savefig('feature_imp2-ai.pdf')
plt.savefig('feature_imp3-ai.svg')
