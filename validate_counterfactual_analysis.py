import numpy as np
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import collections
from scipy import stats
from lib.treatments import hypertension_treatments, dm_treatment, ascvd_treatment, hypertension_treatments_history, dm_treatment_history
from sklearn.neighbors import BallTree
from ascvd_risk import framingham_ascvd_risk

diagnosis_reward = False
EPISODES = 20000

target_treatment = None
action_size = 100

def hash_to_action(x):
    return int(''.join(map(str, x)))
    # return sum([int(x[i]) * (2**i) for i in range(len(x))])

# define treatment category globally
treatment_ctg = {'hypertension': hypertension_treatments, 'diabetes': dm_treatment, 'ascvd': ascvd_treatment}


targets = ['ace_inhibitor-thiazide_or_thiazide-like_diuretic',
           'alpha/beta-adrenergic_blocking_agents',
           'angioten.receptr_antag-calcium_chanl_blkr-thiazide',
           'angiotensin_recept-neprilysin_inhibitor_comb(arni)',
           'angiotensin_receptor_antag.-thiazide_diuretic_comb',
           'angiotensin_receptor_blockr-calcium_channel_blockr',
           'anti-inflammatory,_interleukin-1_beta_blockers',
           'antihypergly,dpp-4_enzyme_inhib.-thiazolidinedione',
           'antihypergly,incretin_mimetic(glp-1_recep.agonist)',
           'antihypergly,insulin,long_act-glp-1_recept.agonist',
           'antihyperglycemc-sod/gluc_cotransport2(sglt2)inhib',
           'antihyperglycemic_-_dopamine_receptor_agonists',
           'antihyperglycemic,_alpha-glucosidase_inhibitors',
           'antihyperglycemic,_amylin_analog-type',
           'antihyperglycemic,_biguanide_type',
           'antihyperglycemic,_dpp-4_inhibitors',
           'antihyperglycemic,_insulin-release_stimulant_type',
           'antihyperglycemic,_sglt-2_and_dpp-4_inhibitor_comb',
           'antihyperglycemic,_thiazolidinedione_and_biguanide',
           'antihyperglycemic,_thiazolidinedione-sulfonylurea',
           'antihyperglycemic,dpp-4_inhibitor-biguanide_combs.',
           'antihyperglycemic,insulin-release_stim.-biguanide',
           'antihyperglycemic,thiazolidinedione(pparg_agonist)',
           'antihyperglycemic-sglt2_inhibitor-biguanide_combs.',
           'antihyperlip.hmg_coa_reduct_inhib-cholest.ab.inhib',
           'antihyperlipid-_hmg-coa_ri-calcium_channel_blocker',
           'antihyperlipidemic_-_hmg_coa_reductase_inhibitors',
           'antihyperlipidemic_-_pcsk9_inhibitors',
           'antihyperlipidemic-hmg_coa_reductase_inhib.-niacin',
           'antihypertensives,_ace_inhibitors',
           'antihypertensives,_angiotensin_receptor_antagonist',
           'beta-adrenergic_blocking_agents',
           'beta-blockers_and_thiazide,thiazide-like_diuretics',
           'bile_salt_sequestrants',
           'bulk_chemicals',
           'calcium_channel_blocking_agents',
           'insulins',
           'lipotropics',
           'miotics_and_other_intraocular_pressure_reducers',
           'potassium_sparing_diuretics_in_combination',
           'renin_inhibitor,direct_and_thiazide_diuretic_comb',
           'renin_inhibitor,direct-angiotensin_receptr_antagon',
           'thiazide_and_related_diuretics']

excluded = ['study_id', 'encounter_dt_ran']

data_path = '~/Research/PHD/project/Hua Zheng/previous code/cleaned_EHR_treatment_param_lab_test_final_3diseases-v6.csv'
# '~/Research/PHD/project/Hua Zheng/previous code/cleaned_EHR_treatment_param_lab_test_final_3diseases_cvd_encounter_diagnosis.csv'
data = pd.read_csv(data_path)
if diagnosis_reward:
    data['reward'] = data['reward_diagnosis']
    data.drop('reward_diagnosis', inplace=True, axis=1)

# %% weight different rewards
# reward_weight = [1, 0, 0]
# data['reward'] = data.apply(lambda x: (x['reward_bp'] * reward_weight[0] + x['reward_ascvd'] * reward_weight[1] + x[
#     'reward_diabetes'] * reward_weight[2]) / sum(reward_weight),
#                             axis=1)  # (data[['reward_bp']] * 2 + data[['reward_ascvd']] + data[['reward_ascvd']] * 2) / 5#, 'reward_ascvd', 'reward_diabetes'
# data = data.drop(['egfr_mdrd_african_american_min', 'egfr_mdrd_african_american_max', 'egfr_mdrd_african_american', 'egfr_mdrd_non_african_american', 'egfr_mdrd_non_african_american_max','egfr_mdrd_non_african_american_min', 'next_egfr_mdrd_african_american','next_egfr_mdrd_african_american_max','next_egfr_mdrd_african_american_min', 'next_egfr_mdrd_non_african_american','next_egfr_mdrd_non_african_american_max', 'next_egfr_mdrd_non_african_american_min', 'bulk_chemicals_hist', 'next_bulk_chemicals_hist'],axis=1)
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
# if target_treatment != None:
#     label_cols = treatment_ctg[target_treatment]
#     action_size = len(label_cols)

target = data[label_cols].apply(lambda x: hash_to_action(x), axis=1)
data['target'] = target
counter = collections.Counter(target)
target_set = set([i[0] for i in counter.most_common(action_size)])
target_replacement = dict(zip(iter(target_set), range(action_size)))
target_column_renames = ['target' + str(i) for i in range(action_size)]

data = data[data.target.apply(lambda x: x in target_set)]
data['target'] = data['target'].replace(target_replacement)
reward_cols = ['reward']

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
                   'next_bp_systolic_min',
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
                   'next_bp_diastolic_max']
state_cols = list(set(data.columns) - set(full_label_cols) - {'target'} - set(
    ['reward', 'reward_bp', 'reward_ascvd', 'reward_diabetes', 'risk_ascvd', 'next_risk_ascvd']) - set(
    ['study_id', 'encounter_dt_ran']) - set(
    next_state_cols))
state_cols = [s[5:] for s in next_state_cols]
if diagnosis_reward:
    state_cols = state_cols - (['CVD', 'days_to_CVD'])

# next_state_cols = ['next_' + s for s in list(state_cols)]
# patients_column = data['study_id']
data = data.drop(excluded, axis=1)

# data = data.drop(excluded, axis=1)
_temp = data.drop('target', axis=1).max(skipna=True) - data.drop('target', axis=1).min(skipna=True)
_temp[data.drop('target', axis=1).columns[(_temp == 0).values]] = 1.0
normalized_df = (data.drop(['target'], axis=1) - data.drop('target', axis=1).min(skipna=True)) / _temp

# constant_variable_df = data.drop('target', axis=1).max(skipna=True) - data.drop('target', axis=1).min(skipna=True)
# dropped_state = normalized_df.columns[(constant_variable_df == 0).values]
# normalized_df[dropped_state] = data[dropped_state]
normalized_df['reward'] = (data['reward'] - data['reward'].min(skipna=True)) / data['reward'].max()
# next_state_cols = set(next_state_cols) - set(dropped_state)
# state_cols = set(state_cols) - set(dropped_state)
ohe = to_categorical(data['target'], action_size)

for i, col in enumerate(target_column_renames):
    normalized_df[col] = ohe[:, i]


train, test = train_test_split(normalized_df, test_size=0.4, random_state=2019)
interested_train = train  # train[train[targets].apply(lambda x: sum(x) > 0, axis=1)]
interested_tests = test  # test[test[targets].apply(lambda x: sum(x) > 0, axis=1)]

from sklearn.decomposition import PCA

# pca = PCA(n_components=8)
num_neighbors = 8
pca = PCA(.85)
pca.fit(interested_tests[state_cols])
principalComponents = pca.transform(interested_tests[state_cols])
num_pc = principalComponents.shape[1]
#interested_tests = interested_tests.join(data['target'])
#y = np.append(interested_tests['target'].values.reshape([interested_tests.shape[0], 1]), axis=1)
principalComponents = np.append(principalComponents[:, :num_pc], interested_tests['target'].values.reshape([interested_tests.shape[0], 1]), axis=1)

principalComponents[:,-1] = principalComponents[:,-1] * 10000
tree = BallTree(principalComponents[:, :num_pc + 1], leaf_size=1000, metric='euclidean')
# nbrs = tree.query(principalComponents.values[unmatched_indices][:,list(range(21)) + [22]], k=10, return_distance=False)
nbrs = tree.query(principalComponents[:, list(range(num_pc+1))], k=num_neighbors, return_distance=False)
# from joblib import dump, load


interested_tests['age_raw'] = interested_tests['age'] * (
        data['age'].max() - data['age'].min()) + data['age'].min()
# interested_tests['next_sex_raw'] = interested_tests['next_sex_male']

# interested_tests['next_smoke']

interested_tests['has_hypertension'] = interested_tests[hypertension_treatments_history].sum(axis=1) > 0
interested_tests['has_diabetes'] = interested_tests[dm_treatment_history].sum(axis=1) > 0

interested_tests['next_bp_systolic_raw'] = interested_tests['next_bp_systolic'] * (
        data['next_bp_systolic'].max() - data['next_bp_systolic'].min()) + data['next_bp_systolic'].min()
interested_tests['next_bp_diastolic_raw'] = interested_tests['next_bp_diastolic'] * (
        data['next_bp_diastolic'].max() - data['next_bp_diastolic'].min()) + data['next_bp_diastolic'].min()
interested_tests['next_hemoglobin_a1c_raw'] = interested_tests['next_hemoglobin_a1c'] * (
        data['next_hemoglobin_a1c'].max() - data['next_hemoglobin_a1c'].min()) + data[
                                                  'next_hemoglobin_a1c'].min()
interested_tests['next_risk_ascvd_raw'] = interested_tests['next_risk_ascvd'] * (
        data['next_risk_ascvd'].max() - data['next_risk_ascvd'].min()) + data['next_risk_ascvd'].min()
interested_tests['reward_raw'] = interested_tests['reward'] * (data['reward'].max() - data['reward'].min()) + data[
    'reward'].min()
# interested_tests['reward_ascvd_raw'] = interested_tests['reward_ascvd'] * (
#             data['reward_ascvd'].max() - data['reward_ascvd'].min()) + data['reward_ascvd'].min()

interested_tests['next_triglycerides_raw'] = interested_tests['next_triglycerides'] * (
        data['next_triglycerides'].max() - data['next_triglycerides'].min()) + data['next_triglycerides'].min()
interested_tests['next_cholesterol_total_raw'] = interested_tests['next_cholesterol,_total'] * (
        data['next_cholesterol,_total'].max() - data['next_cholesterol,_total'].min()) + data[
                                                     'next_cholesterol,_total'].min()
interested_tests['next_ldl_cholesterol_raw'] = interested_tests['next_ldl_cholesterol'] * (
        data['next_ldl_cholesterol'].max() - data['next_ldl_cholesterol'].min()) + data['next_ldl_cholesterol'].min()

interested_tests['next_hdl_cholesterol_raw'] = interested_tests['next_hdl_cholesterol'] * (
        data['next_hdl_cholesterol'].max() - data['next_hdl_cholesterol'].min()) + data['next_hdl_cholesterol'].min()
interested_tests.index = np.arange(0, len(interested_tests))
test_bp_dict = interested_tests[
    ['age_raw', 'next_smoke', 'sex_male', 'has_hypertension', 'has_diabetes', 'next_bp_systolic_raw',
     'next_bp_diastolic_raw', 'next_triglycerides_raw', 'next_cholesterol_total_raw', 'next_hdl_cholesterol_raw',
     'next_risk_ascvd_raw', 'next_hemoglobin_a1c_raw', 'next_ldl_cholesterol_raw',
     'reward_raw']].to_dict('index')

# ONLY_UNMATCHED = True
# if ONLY_UNMATCHED:
#     temp_df = interested_tests[unmatched_indices.tolist()]
#     # [['next_bp_systolic_raw', 'next_bp_diastolic_raw']]
#     temp_df.index = np.arange(0, len(temp_df))
#     test_bp_dict = temp_df[['next_bp_systolic_raw', 'next_bp_diastolic_raw', 'next_risk_ascvd_raw', 'next_hemoglobin_a1c_raw','reward_raw']].to_dict('index')

# validate
gain_bp_systolic = []
gain_bp_diastolic = []
gain_triglycerides = []
gain_cholesterol_total = []
gain_hdl_cholesterol = []
gain_ldl_cholesterol = []
gain_risk_ascvd = []
gain_hemoglobin_a1c = []
gain_reward = []

RL_bp_systolic = []
clinician_bp_systolic = []
RL_bp_diastolic = []
clinician_bp_diastolic = []
RL_triglycerides = []
clinician_triglycerides = []
RL_cholesterol_total = []
clinician_cholesterol_total = []
RL_hdl_cholesterol = []
clinician_hdl_cholesterol = []
RL_ldl_cholesterol = []
clinician_ldl_cholesterol = []
RL_hemoglobin_a1c = []
clinician_hemoglobin_a1c = []
RL_reward = []
clinician_reward = []
RL_risk_ascvd = []
clinician_risk_ascvd = []
doctor_reward = []

ONLY_UNMATCHED = False
if ONLY_UNMATCHED:
    test_neighbor = nbrs[unmatched_indices.tolist()]
    neighbors = zip([i for i, x in enumerate(unmatched_indices.tolist()) if x],
                    test_neighbor)  # zip(unmatched_indices.tolist())
else:
    test_neighbor = nbrs
    neighbors = zip(range(test_neighbor.shape[0]), test_neighbor)

for idx, v in list(neighbors):
    ii = 0
    bp_systolic = .0
    bp_diastolic = .0
    triglycerides = .0
    cholesterol_total = .0
    hdl_cholesterol = .0
    ldl_cholesterol = .0
    # cholesterol_total_list = [.0]*num_neighbors
    # hdl_cholesterol_list = [.0]*num_neighbors
    risk_ascvd = [.0] * num_neighbors
    hemoglobin_a1c = .0
    reward_ = .0
    avg_risk_ascvd = .0
    # risk_ascvd_sum = .0
    for k in v:
        bp_systolic += test_bp_dict[k]['next_bp_systolic_raw']
        bp_diastolic += test_bp_dict[k]['next_bp_diastolic_raw']
        triglycerides += test_bp_dict[k]['next_triglycerides_raw']
        cholesterol_total += test_bp_dict[k]['next_cholesterol_total_raw']
        hdl_cholesterol += test_bp_dict[k]['next_hdl_cholesterol_raw']
        ldl_cholesterol += test_bp_dict[k]['next_ldl_cholesterol_raw']
        avg_risk_ascvd += test_bp_dict[k]['next_risk_ascvd_raw']
        hemoglobin_a1c += test_bp_dict[k]['next_hemoglobin_a1c_raw']
        reward_ += test_bp_dict[k]['reward_raw']
        # cholesterol_total_list[ii], hdl_cholesterol_list[ii]=test_bp_dict[k]['next_cholesterol_total_raw'], test_bp_dict[k]['next_hdl_cholesterol_raw']
        risk_ascvd[ii] = framingham_ascvd_risk([test_bp_dict[idx]['sex_male'],
                                                test_bp_dict[idx]['age_raw'],
                                                test_bp_dict[k]['next_cholesterol_total_raw'],
                                                test_bp_dict[k]['next_hdl_cholesterol_raw'],
                                                test_bp_dict[k]['next_bp_systolic_raw'],
                                                test_bp_dict[idx]['next_smoke'],
                                                test_bp_dict[idx]['has_hypertension'],
                                                test_bp_dict[idx]['has_diabetes']])
        ii += 1
    # cholesterol_total_list.remove(min(cholesterol_total_list))
    # #cholesterol_total_list.remove(max(cholesterol_total_list))
    # hdl_cholesterol_list.remove(min(hdl_cholesterol_list))
    # #hdl_cholesterol_list.remove(max(hdl_cholesterol_list))
    # risk_ascvd_sum = framingham_ascvd_risk([test_bp_dict[idx]['sex_male'],
    #                                         test_bp_dict[idx]['age_raw'],
    #                                         np.mean(cholesterol_total_list),
    #                                         np.mean(hdl_cholesterol_list),
    #                                         bp_systolic / num_neighbors,
    #                                         test_bp_dict[idx]['next_smoke'],
    #                                         test_bp_dict[idx]['has_hypertension'],
    #                                         test_bp_dict[idx]['has_diabetes']])
    # risk_ascvd.remove(min(risk_ascvd))
    # risk_ascvd.remove(min(risk_ascvd))
    # risk_ascvd.remove(max(risk_ascvd))

    gain_bp_systolic.append(bp_systolic / num_neighbors - test_bp_dict[idx]['next_bp_systolic_raw'])
    gain_bp_diastolic.append(bp_diastolic / num_neighbors - test_bp_dict[idx]['next_bp_diastolic_raw'])
    gain_triglycerides.append(triglycerides / num_neighbors - test_bp_dict[idx]['next_triglycerides_raw'])
    gain_cholesterol_total.append(cholesterol_total / num_neighbors - test_bp_dict[idx]['next_cholesterol_total_raw'])
    gain_hdl_cholesterol.append(hdl_cholesterol / num_neighbors - test_bp_dict[idx]['next_hdl_cholesterol_raw'])
    gain_ldl_cholesterol.append(ldl_cholesterol / num_neighbors - test_bp_dict[idx]['next_ldl_cholesterol_raw'])
    gain_risk_ascvd.append(avg_risk_ascvd/num_neighbors - test_bp_dict[idx]['next_risk_ascvd_raw'])
    # gain_risk_ascvd.append(np.mean(risk_ascvd) - test_bp_dict[idx]['next_risk_ascvd_raw'])
    gain_hemoglobin_a1c.append(hemoglobin_a1c / num_neighbors - test_bp_dict[idx]['next_hemoglobin_a1c_raw'])
    gain_reward.append(reward_ / num_neighbors - test_bp_dict[idx]['reward_raw'])
    # gain_risk_ascvd.append(risk_ascvd / num_neighbors - test_bp_dict[idx]['reward_ascvd_raw'])

    RL_bp_systolic.append(bp_systolic / num_neighbors)
    clinician_bp_systolic.append(test_bp_dict[idx]['next_bp_systolic_raw'])

    RL_bp_diastolic.append(bp_diastolic / num_neighbors)
    clinician_bp_diastolic.append(test_bp_dict[idx]['next_bp_diastolic_raw'])

    RL_triglycerides.append(triglycerides / num_neighbors)
    clinician_triglycerides.append(test_bp_dict[idx]['next_triglycerides_raw'])

    RL_cholesterol_total.append(cholesterol_total / num_neighbors)
    clinician_cholesterol_total.append(test_bp_dict[idx]['next_cholesterol_total_raw'])

    RL_hdl_cholesterol.append(hdl_cholesterol / num_neighbors)
    clinician_hdl_cholesterol.append(test_bp_dict[idx]['next_hdl_cholesterol_raw'])

    RL_ldl_cholesterol.append(ldl_cholesterol / num_neighbors)
    clinician_ldl_cholesterol.append(test_bp_dict[idx]['next_ldl_cholesterol_raw'])

    # RL_risk_ascvd.append(avg_risk_ascvd / num_neighbors)
    RL_risk_ascvd.append(np.mean(risk_ascvd))
    clinician_risk_ascvd.append(test_bp_dict[idx]['next_risk_ascvd_raw'])

    RL_hemoglobin_a1c.append(hemoglobin_a1c / num_neighbors)
    clinician_hemoglobin_a1c.append(test_bp_dict[idx]['next_hemoglobin_a1c_raw'])

    RL_reward.append(reward_ / num_neighbors)
    clinician_reward.append(test_bp_dict[idx]['reward_raw'])

    # RL_risk_ascvd.append(risk_ascvd / num_neighbors)
    # clinician_risk_ascvd.append(test_bp_dict[idx]['reward_ascvd_raw'])

print('-----------------------------')
print('--Comparison: RL - Doctor --')
print("bp_systolic: {0:.4f}(SD:{1:.4f},SE:{2:.4f},pvalue:{3:.4f},cor:{4:.4f}))".format(np.mean(gain_bp_systolic),
                                                                           np.std(gain_bp_systolic),
                                                                           np.std(gain_bp_systolic) / np.sqrt(
                                                                               len(gain_bp_systolic)),
                                                                           stats.ttest_ind(RL_bp_systolic,
                                                                                           clinician_bp_systolic,
                                                                                           equal_var=False)[1],
                                                                           stats.pearsonr(RL_bp_systolic,clinician_bp_systolic)[0]))
print("bp_diastolic: {0:.4f}(SD:{1:.4f},SE:{2:.4f},pvalue:{3:.4f},cor:{4:.4f}))".format(np.mean(gain_bp_diastolic),
                                                                            np.std(gain_bp_diastolic),
                                                                            np.std(gain_bp_diastolic) / np.sqrt(
                                                                                len(gain_bp_diastolic)),
                                                                            stats.ttest_ind(RL_bp_diastolic,
                                                                                            clinician_bp_diastolic,
                                                                                            equal_var=False)[1],stats.pearsonr(RL_bp_diastolic,clinician_bp_diastolic)[0]))
print("triglycerides: {0:.4f}(SD:{1:.4f},SE:{2:.4f},pvalue:{3:.4f},cor:{4:.4f}))".format(np.mean(gain_triglycerides),
                                                                             np.std(gain_triglycerides),
                                                                             np.std(gain_triglycerides) / np.sqrt(
                                                                                 len(gain_triglycerides)),
                                                                             stats.ttest_ind(RL_triglycerides,
                                                                                             clinician_triglycerides,
                                                                                             equal_var=False)[1],stats.pearsonr(RL_triglycerides,clinician_triglycerides)[0]))
print("cholesterol_total: {0:.4f}(SD:{1:.4f},SE:{2:.4f},pvalue:{3:.4f},cor:{4:.4f}))".format(np.mean(gain_cholesterol_total),
                                                                                 np.std(gain_cholesterol_total), np.std(
        gain_cholesterol_total) / np.sqrt(len(gain_cholesterol_total)), stats.ttest_ind(RL_cholesterol_total,
                                                                                        clinician_cholesterol_total,
                                                                                        equal_var=False)[1],stats.pearsonr(RL_cholesterol_total,clinician_cholesterol_total)[0]))
print("hdl cholesterol: {0:.4f}(SD:{1:.4f},SE:{2:.4f},pvalue:{3:.4f},cor:{4:.4f}))".format(np.mean(gain_hdl_cholesterol),
                                                                               np.std(gain_hdl_cholesterol),
                                                                               np.std(gain_hdl_cholesterol) / np.sqrt(
                                                                                   len(gain_hdl_cholesterol)),
                                                                               stats.ttest_ind(RL_hdl_cholesterol,
                                                                                               clinician_hdl_cholesterol,
                                                                                               equal_var=False)[1],stats.pearsonr(RL_hdl_cholesterol,clinician_hdl_cholesterol)[0]))
print("ldl cholesterol: {0:.4f}(SD:{1:.4f},SE:{2:.4f},pvalue:{3:.4f},cor:{4:.4f}))".format(np.mean(gain_ldl_cholesterol),
                                                                               np.std(gain_ldl_cholesterol),
                                                                               np.std(gain_ldl_cholesterol) / np.sqrt(
                                                                                   len(gain_ldl_cholesterol)),
                                                                               stats.ttest_ind(RL_ldl_cholesterol,
                                                                                               clinician_ldl_cholesterol,
                                                                                               equal_var=False)[1],stats.pearsonr(RL_ldl_cholesterol,clinician_ldl_cholesterol)[0]))
print("risk_ascvd: {0:.4f}(SD:{1:.4f},SE:{2:.4f},pvalue:{3:.4f},cor:{4:.4f}))".format(np.mean(gain_risk_ascvd),
                                                                          np.std(gain_risk_ascvd),
                                                                          np.std(gain_risk_ascvd) / np.sqrt(
                                                                              len(gain_risk_ascvd)),
                                                                          stats.ttest_ind(RL_risk_ascvd,
                                                                                          clinician_risk_ascvd,
                                                                                          equal_var=False)[1],stats.pearsonr(RL_risk_ascvd,clinician_risk_ascvd)[0]))
print("hemoglobin_a1c: {0:.4f}(SD:{1:.4f},SE:{2:.4f},pvalue:{3:.4f},cor:{4:.4f}))".format(np.mean(gain_hemoglobin_a1c),
                                                                              np.std(gain_hemoglobin_a1c),
                                                                              np.std(gain_hemoglobin_a1c) / np.sqrt(
                                                                                  len(gain_hemoglobin_a1c)),
                                                                              stats.ttest_ind(RL_hemoglobin_a1c,
                                                                                              clinician_hemoglobin_a1c,
                                                                                              equal_var=False)[1],stats.pearsonr(RL_hemoglobin_a1c,clinician_hemoglobin_a1c)[0]))
print("reward: {0:.4f}(SD:{1:.4f},SE:{2:.4f},pvalue:{3:.4f},cor:{4:.4f}))".format(np.mean(gain_reward), np.std(gain_reward),
                                                                      np.std(gain_reward) / np.sqrt(len(gain_reward)),
                                                                      stats.ttest_ind(RL_reward, clinician_reward,
                                                                                      equal_var=False)[1],stats.pearsonr(RL_reward,clinician_reward)[0]))

print("RL bp_systolic: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(RL_bp_systolic), np.std(RL_bp_systolic),
                                                              np.std(RL_bp_systolic) / np.sqrt(len(RL_bp_systolic))))
print("RL bp_diastolic: {0:.4f}(SD:{1:.4f},SE:{2:.4f}))".format(np.mean(RL_bp_diastolic), np.std(RL_bp_diastolic),
                                                                np.std(RL_bp_diastolic) / np.sqrt(
                                                                    len(RL_bp_diastolic))))
print("RL triglycerides: {0:.4f}(SD:{1:.4f},SE:{2:.4f}))".format(np.mean(RL_triglycerides), np.std(RL_triglycerides),
                                                                 np.std(RL_triglycerides) / np.sqrt(
                                                                     len(RL_triglycerides))))
print("RL cholesterol_total: {0:.4f}(SD:{1:.4f},SE:{2:.4f}))".format(np.mean(RL_cholesterol_total),
                                                                     np.std(RL_cholesterol_total),
                                                                     np.std(RL_cholesterol_total) / np.sqrt(
                                                                         len(RL_cholesterol_total))))
print("RL hdl cholesterol: {0:.4f}(SD:{1:.4f},SE:{2:.4f}))".format(np.mean(RL_hdl_cholesterol),
                                                                   np.std(RL_hdl_cholesterol),
                                                                   np.std(RL_hdl_cholesterol) / np.sqrt(
                                                                       len(RL_hdl_cholesterol))))
print("RL ldl cholesterol: {0:.4f}(SD:{1:.4f},SE:{2:.4f}))".format(np.mean(RL_ldl_cholesterol),
                                                                   np.std(RL_ldl_cholesterol),
                                                                   np.std(RL_ldl_cholesterol) / np.sqrt(
                                                                       len(RL_ldl_cholesterol))))
print("RL risk_ascvd: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(RL_risk_ascvd), np.std(RL_risk_ascvd),
                                                             np.std(RL_risk_ascvd) / np.sqrt(len(RL_risk_ascvd))))
print("RL hemoglobin_a1c: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(RL_hemoglobin_a1c), np.std(RL_hemoglobin_a1c),
                                                                 np.std(RL_hemoglobin_a1c) / np.sqrt(
                                                                     len(RL_hemoglobin_a1c))))
print("RL reward: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(RL_reward), np.std(RL_reward),
                                                         np.std(RL_reward) / np.sqrt(len(RL_reward))))

print("clinician bp_systolic: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(clinician_bp_systolic),
                                                                     np.std(clinician_bp_systolic),
                                                                     np.std(clinician_bp_systolic) / np.sqrt(
                                                                         len(clinician_bp_systolic))))
print("clinician bp_diastolic: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(clinician_bp_diastolic),
                                                                      np.std(clinician_bp_diastolic),
                                                                      np.std(clinician_bp_diastolic) / np.sqrt(
                                                                          len(clinician_bp_diastolic))))
print("clinician triglycerides: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(clinician_triglycerides),
                                                                       np.std(clinician_triglycerides),
                                                                       np.std(clinician_triglycerides) / np.sqrt(
                                                                           len(clinician_triglycerides))))
print("clinician cholesterol_total: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(clinician_cholesterol_total),
                                                                           np.std(clinician_cholesterol_total), np.std(
        clinician_cholesterol_total) / np.sqrt(len(clinician_cholesterol_total))))
print("clinician hdl cholesterol: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(clinician_hdl_cholesterol),
                                                                         np.std(clinician_hdl_cholesterol),
                                                                         np.std(clinician_hdl_cholesterol) / np.sqrt(
                                                                             len(clinician_hdl_cholesterol))))
print("clinician ldl cholesterol: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(clinician_ldl_cholesterol),
                                                                         np.std(clinician_ldl_cholesterol),
                                                                         np.std(clinician_ldl_cholesterol) / np.sqrt(
                                                                             len(clinician_ldl_cholesterol))))
print("clinician risk_ascvd: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(clinician_risk_ascvd),
                                                                    np.std(clinician_risk_ascvd),
                                                                    np.std(clinician_risk_ascvd) / np.sqrt(
                                                                        len(clinician_risk_ascvd))))
print("clinician hemoglobin_a1c: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(clinician_hemoglobin_a1c),
                                                                        np.std(clinician_hemoglobin_a1c),
                                                                        np.std(clinician_hemoglobin_a1c) / np.sqrt(
                                                                            len(clinician_hemoglobin_a1c))))
print("clinician reward: {0:.4f}(SD:{1:.4f},SE:{2:.4f})".format(np.mean(clinician_reward), np.std(clinician_reward),
                                                                np.std(clinician_reward) / np.sqrt(
                                                                    len(clinician_reward))))
