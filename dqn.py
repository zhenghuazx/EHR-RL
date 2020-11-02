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
import collections
from lib.treatments import hypertension_treatments, dm_treatment, ascvd_treatment, hypertension_treatments_history, dm_treatment_history
from ascvd_risk import framingham_ascvd_risk
from lib.DQN import DQNAgent
diagnosis_reward = False
EPISODES = 16000


def hash_to_action(x):
    return int(''.join(map(str, x)))
    # return sum([int(x[i]) * (2**i) for i in range(len(x))])

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def decode_to_treatment(x, labels):
    gap = len(labels) - len(x)
    index_temp = findOccurrences(x, '1')
    output = [labels[idx + gap] for idx in index_temp]
    return output

# define treatment category globally
treatment_ctg = {'hypertension': hypertension_treatments, 'diabetes': dm_treatment, 'ascvd': ascvd_treatment}

all_treamments = hypertension_treatments.union(dm_treatment).union(ascvd_treatment)
cur_next_treatments = set(["next_" + i for i in all_treamments]).union(all_treamments)

def map_to_disease_category(x, disease, label_cols):
    for t in range(len(x)):
        if x[t] == 1 and label_cols[t] in treatment_ctg[disease]:
            return True
    return False

def loss_weight(data):
    loss_weights = {}
    for i in data[['target']].values:
        if i[0] in loss_weights:
            loss_weights[i[0]] = loss_weights[i[0]] + 1
        else:
            loss_weights[i[0]] = 1

    return dict([[str(k), v / data[['target']].shape[0]] for k, v in loss_weights.items()])

if __name__ == "__main__":
    # Select the target disease('hypertension', 'diabetes', 'ascvd'), if use "None" represent multimorbidity
    target_treatment = None
    action_size = 100

    excluded = ['study_id', 'encounter_dt_ran']

    data_path = '~/Research/PHD/project/Hua Zheng/previous code/cleaned_EHR_treatment_param_lab_test_final_3diseases-v6.csv'
    data = pd.read_csv(data_path)
    if diagnosis_reward:
        data['reward'] = data['reward_diagnosis']
        data.drop('reward_diagnosis', inplace=True, axis=1)

    #%% weight different rewards
    reward_weight = None
    if target_treatment == 'hypertension':
        reward_weight = [1, 0, 0]
    if target_treatment == 'diabetes':
        reward_weight = [0, 1, 0]
    if target_treatment == 'ascvd':
        reward_weight = [0, 0, 1]
    data['reward'] = data.apply(lambda x: (x['reward_bp'] * reward_weight[0] + x['reward_ascvd'] * reward_weight[1] + x['reward_diabetes'] * reward_weight[2]) / sum(reward_weight), axis = 1) # (data[['reward_bp']] * 2 + data[['reward_ascvd']] + data[['reward_ascvd']] * 2) / 5#, 'reward_ascvd', 'reward_diabetes'
#    data = data.drop(['egfr_mdrd_african_american_min', 'egfr_mdrd_african_american_max', 'egfr_mdrd_african_american', 'egfr_mdrd_non_african_american', 'egfr_mdrd_non_african_american_max','egfr_mdrd_non_african_american_min', 'next_egfr_mdrd_african_american','next_egfr_mdrd_african_american_max','next_egfr_mdrd_african_american_min', 'next_egfr_mdrd_non_african_american','next_egfr_mdrd_non_african_american_max', 'next_egfr_mdrd_non_african_american_min', 'bulk_chemicals_hist', 'next_bulk_chemicals_hist'],axis=1)
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
    # select visits that are only target_treatment
    # visits_to_remove = data[['study_id', 'encounter_dt_ran']][data.target.apply(lambda x: x not in target_set)]
    # merged = data[['study_id', 'encounter_dt_ran']].merge(visits_to_remove, indicator=True, how='outer')
    # merged = merged[merged['_merge'] == 'left_only']

    data = data[data.target.apply(lambda x: x in target_set)]
    data['target'] = data['target'].replace(target_replacement)
    reward_cols = ['reward']

    # %% generate the association between diseases and aggregated actions
    # temp_dat = data[:100000][['target', 'reward']]
    # temp_dat['hypertension'] = data[label_cols][:100000].apply(lambda x: map_to_disease_category(x, 'hypertension', label_cols), axis=1)
    # temp_dat['diabetes'] = data[label_cols][:100000].apply(lambda x: map_to_disease_category(x, 'diabetes', label_cols), axis=1)
    # temp_dat['ascvd'] = data[label_cols][:100000].apply(lambda x: map_to_disease_category(x, 'ascvd', label_cols), axis=1)
    # temp_dat.drop('reward', axis=1).drop_duplicates().count()



    # %%
    next_state_cols = ['next_creatinine',
                       'next_bmi',
                       'next_hemoglobin_a1c',
                       'next_antihyperlipidemic-hmg_coa_reductase_inhib.-niacin_hist',
                       'next_hemoglobin_a1c_max',
                       'next_bp_diastolic_min',
                       'next_antihyperglycemic,_sglt-2_and_dpp-4_inhibitor_comb_cur',
                       'next_antihyperglycemic,_dpp-4_inhibitors_hist',
                       'next_race_white',
                       'next_antihyperglycemc-sod/gluc_cotransport2(sglt2)inhib_hist',
                       'next_ldl_cholesterol_min',
                       'next_antihyperlip.hmg_coa_reduct_inhib-cholest.ab.inhib_cur',
                       # 'next_egfr_mdrd_non_african_american_min',
                       'next_alpha/beta-adrenergic_blocking_agents_cur',
                       'next_antihyperglycemic,insulin-release_stim.-biguanide_cur',
                       'next_antihypergly,insulin,long_act-glp-1_recept.agonist_cur',
                       'next_angioten.receptr_antag-calcium_chanl_blkr-thiazide_cur',
                       'next_anti-inflammatory,_interleukin-1_beta_blockers_cur',
                       'next_beta-blockers_and_thiazide,thiazide-like_diuretics_hist',
                       'next_antihyperglycemic,_thiazolidinedione_and_biguanide_hist',
                       'next_antihypertensives,_ace_inhibitors_cur',
                       # 'next_egfr_mdrd_african_american',
                       # 'next_egfr_mdrd_non_african_american',
                       'next_race_native_hawaiian_or_other_pacific_islander',
                       'next_antihypertensives,_angiotensin_receptor_antagonist_hist',
                       'next_antihyperglycemic,_insulin-release_stimulant_type_cur',
                       'next_antihypergly,incretin_mimetic(glp-1_recep.agonist)_cur',
                       'next_antihyperglycemic,insulin-release_stim.-biguanide_hist',
                       'next_angiotensin_receptor_blockr-calcium_channel_blockr_hist',
                       'next_antihyperglycemic,_biguanide_type_cur',
                       'next_antihyperglycemic-sglt2_inhibitor-biguanide_combs._cur',
                       'next_creatinine_min',
                       'next_bmi_min',
                       # 'next_egfr_mdrd_african_american_max',
                       'next_antihyperglycemic_-_dopamine_receptor_agonists_cur',
                       'next_beta-adrenergic_blocking_agents_hist',
                       'next_antihyperglycemic_-_dopamine_receptor_agonists_hist',
                       'next_insulins_cur',
                       'next_race_native_american',
                       'next_hdl_cholesterol',
                       'next_race_multiple_race',
                       'next_antihyperglycemic,_sglt-2_and_dpp-4_inhibitor_comb_hist',
                       'next_antihyperlipidemic_-_hmg_coa_reductase_inhibitors_cur',
                       'next_renin_inhibitor,direct-angiotensin_receptr_antagon_hist',
                       'next_anti-inflammatory,_interleukin-1_beta_blockers_hist',
                       'next_angioten.receptr_antag-calcium_chanl_blkr-thiazide_hist',
                       'next_antihyperlipidemic-hmg_coa_reductase_inhib.-niacin_cur',
                       'next_triglycerides_max',
                       'next_antihyperglycemic-sglt2_inhibitor-biguanide_combs._hist',
                       'next_hemoglobin_a1c_min',
                       'next_triglycerides',
                       # 'next_egfr_mdrd_non_african_american_max',
                       'next_miotics_and_other_intraocular_pressure_reducers_hist',
                       'next_potassium_sparing_diuretics_in_combination_cur',
                       'next_bmi_max',
                       'next_beta-adrenergic_blocking_agents_cur',
                       'next_race_other',
                       'next_time_last_vist',
                       'next_antihyperglycemic,_thiazolidinedione_and_biguanide_cur',
                       'next_bp_systolic',
                       'next_bile_salt_sequestrants_cur',
                       'next_antihyperglycemic,dpp-4_inhibitor-biguanide_combs._cur',
                       'next_angiotensin_recept-neprilysin_inhibitor_comb(arni)_cur',
                       'next_race_patient_refused',
                       'next_antihyperglycemic,_amylin_analog-type_cur',
                       'next_antihyperglycemic,_alpha-glucosidase_inhibitors_cur',
                       'next_antihypergly,dpp-4_enzyme_inhib.-thiazolidinedione_cur',
                       'next_renin_inhibitor,direct_and_thiazide_diuretic_comb_cur',
                       'next_angiotensin_receptor_antag.-thiazide_diuretic_comb_hist',
                       'next_antihyperglycemic,thiazolidinedione(pparg_agonist)_cur',
                       'next_cholesterol,_total',
                       'next_antihyperlipid-_hmg-coa_ri-calcium_channel_blocker_cur',
                       'next_thiazide_and_related_diuretics_hist',
                       'next_calcium_channel_blocking_agents_hist',
                       'next_hdl_cholesterol_max',
                       'next_bulk_chemicals_hist',
                       'next_antihyperlipidemic_-_hmg_coa_reductase_inhibitors_hist',
                       'next_potassium_sparing_diuretics_in_combination_hist',
                       'next_hdl_cholesterol_min',
                       # 'next_egfr_mdrd_african_american_min',
                       'next_race_asian',
                       'next_cholesterol,_total_max',
                       'next_antihyperlipid-_hmg-coa_ri-calcium_channel_blocker_hist',
                       'next_ace_inhibitor-thiazide_or_thiazide-like_diuretic_cur',
                       'next_ace_inhibitor-thiazide_or_thiazide-like_diuretic_hist',
                       'next_antihyperglycemic,_alpha-glucosidase_inhibitors_hist',
                       'next_antihyperglycemic,dpp-4_inhibitor-biguanide_combs._hist',
                       'next_lipotropics_hist',
                       'next_antihypergly,dpp-4_enzyme_inhib.-thiazolidinedione_hist',
                       'next_cholesterol,_total_min',
                       'next_smoke',
                       'next_ldl_cholesterol_max',
                       'next_alpha/beta-adrenergic_blocking_agents_hist',
                       'next_antihyperglycemic,_insulin-release_stimulant_type_hist',
                       'next_antihyperlipidemic_-_pcsk9_inhibitors_hist',
                       'next_antihypertensives,_angiotensin_receptor_antagonist_cur',
                       'next_renin_inhibitor,direct_and_thiazide_diuretic_comb_hist',
                       'next_bulk_chemicals_cur',
                       'next_angiotensin_receptor_antag.-thiazide_diuretic_comb_cur',
                       'next_antihyperglycemc-sod/gluc_cotransport2(sglt2)inhib_cur',
                       'next_antihyperglycemic,thiazolidinedione(pparg_agonist)_hist',
                       'next_bile_salt_sequestrants_hist',
                       # 'next_reward',
                       'next_creatinine_max',
                       'next_renin_inhibitor,direct-angiotensin_receptr_antagon_cur',
                       'next_angiotensin_receptor_blockr-calcium_channel_blockr_cur',
                       'next_antihyperglycemic,_thiazolidinedione-sulfonylurea_cur',
                       'next_antihyperlip.hmg_coa_reduct_inhib-cholest.ab.inhib_hist',
                       'next_antihyperglycemic,_biguanide_type_hist',
                       'next_sex_male',
                       'next_antihyperglycemic,_amylin_analog-type_hist',
                       'next_race_unknown',
                       'next_antihypertensives,_ace_inhibitors_hist',
                       'next_antihyperglycemic,_thiazolidinedione-sulfonylurea_hist',
                       'next_antihyperglycemic,_dpp-4_inhibitors_cur',
                       'next_beta-blockers_and_thiazide,thiazide-like_diuretics_cur',
                       'next_miotics_and_other_intraocular_pressure_reducers_cur',
                       'next_triglycerides_min',
                       'next_bp_diastolic',
                       'next_bp_systolic_max',
                       'next_bp_systolic_min',
                       'next_age',
                       'next_calcium_channel_blocking_agents_cur',
                       'next_antihyperlipidemic_-_pcsk9_inhibitors_cur',
                       'next_ldl_cholesterol',
                       'next_antihypergly,insulin,long_act-glp-1_recept.agonist_hist',
                       'next_insulins_hist',
                       'next_thiazide_and_related_diuretics_cur',
                       'next_angiotensin_recept-neprilysin_inhibitor_comb(arni)_hist',
                       'next_lipotropics_cur',
                       'next_antihypergly,incretin_mimetic(glp-1_recep.agonist)_hist',
                       'next_bp_diastolic_max',
                       'next_time_to_first_visit']
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

    normalized_df['reward'] = (data['reward'] - data['reward'].min(skipna=True)) / (data['reward'].max())
    ohe = to_categorical(data['target'], action_size)

    for i, col in enumerate(target_column_renames):
        normalized_df[col] = ohe[:, i]
    normalized_df[['study_id', 'encounter_dt_ran']] = patients_column
    train, test = train_test_split(normalized_df, test_size=0.4, random_state=2019)
    # train2, test2 = train_test_split(data, test_size=0.4, random_state=2019)

    state_size = len(state_cols)
    action_size = len(target_column_renames)
    # loss_weights = loss_weight(data)
    #class_weights = class_weight.compute_class_weight('balanced',
    #                                                  np.unique(data['target'].values),
    #                                                  data['target'].values)
    agent = DQNAgent(state_size, action_size, target_column_renames, state_cols, reward_cols, next_state_cols)#, class_weights, True)
    #agent.load("./model/3d:256-512-256-episodes:20000--dqn-mse-target_treatment:{}.h5".format(target_treatment))
    done = False
    batch_size = 64
    cur_val_error = 0
    episodes_till_target_update = 100
    interested_train = train  # train[train[targets].apply(lambda x: sum(x) > 0, axis=1)]
    interested_tests = test  # test[test[targets].apply(lambda x: sum(x) > 0, axis=1)]
    sample_patient = False
    for e in range(EPISODES):

        # patients_trajectories = random.sample(patients_set, batch_size)
        if sample_patient:
            patient = random.sample(patients_set, 1)
            minibatch = interested_train[interested_train['study_id'] == patient]
        else:
            minibatch = interested_train.sample(batch_size)

        # minibatch = random.sample(agent.memory, batch_size)
        if e % episodes_till_target_update:
            agent.update_target()

        # for visit in minibatch:
        #     loss = agent.replay(minibatch[visit], True)

        agent.learning_rate = agent.learning_rate * (1 - agent.learning_rate_decay)
        loss = agent.replay(minibatch, True)

        if e % 500 == 0:
            cur_val_error = sum(np.argmax(interested_tests[target_column_renames].values, axis=1) == np.argmax(
                agent.model.predict(interested_tests[state_cols]), axis=1)) / interested_tests.shape[0]
        # Logging training loss every 10 timesteps
        if e % 10 == 0:
            print("episode: {}/{}, loss: {:.7f}, test_acc: {:.4f}"
                  .format(e, EPISODES, loss, cur_val_error))
