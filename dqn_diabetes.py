# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
import collections

diagnosis_reward = True
EPISODES = 20000

class DQNAgent:
    def __init__(self, state_size, action_size, targets, state_cols, reward_cols, next_state_cols):
        self.state_size = state_size
        self.action_size = action_size
        self.targets = targets
        self.state_cols = state_cols
        self.reward_cols = reward_cols
        self.next_state_cols = next_state_cols
        self.hidden_layers = {'layers': [128, 256, 128], 'activation': ['tanh', 'tanh', 'tanh']}
        self.gamma = 0.8    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate_decay = 0.01
        self.learning_rate = 0.0004
        self.model = Sequential()
        self.target_model = Sequential()
        self._build_model()
        self._build_target_model()


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        for i in range(len(self.hidden_layers['layers'])):
            if i is 0:
                self.model.add(Dense(self.hidden_layers['layers'][i], input_dim=self.state_size, activation=self.hidden_layers['activation'][i]))
                self.model.add(Dropout(0.5))
            else:
                self.model.add(Dense(self.hidden_layers['layers'][i], activation=self.hidden_layers['activation'][i]))
                self.model.add(Dropout(0.5))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

    def _build_target_model(self):
        for i in range(len(self.hidden_layers['layers'])):
            if i is 0:
                self.target_model.add(Dense(self.hidden_layers['layers'][i], input_dim=self.state_size, activation=self.hidden_layers['activation'][i]))
                self.model.add(Dropout(0.5))
            else:
                self.target_model.add(Dense(self.hidden_layers['layers'][i], activation=self.hidden_layers['activation'][i]))
                self.model.add(Dropout(0.5))
        self.target_model.add(Dense(self.action_size, activation='linear'))
        self.target_model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    # def remember(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    # def act(self, state):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_size)
    #     act_values = self.model.predict(state)
    #     return np.argmax(act_values[0])  # returns action

    def _hash_action(self, actions):
        return sum([pow(2, i) for i in range(8) if actions[i] == 1])

    def replay(self, minibatch, DDQN):
        states = np.array(minibatch[self.state_cols])
        targets_f = self.model.predict(states)
        for idx in range(minibatch.shape[0]):

            next_state = np.reshape(minibatch[self.next_state_cols].iloc[idx].tolist(), [1, state_size])
            reward = minibatch[self.reward_cols].iloc[idx].values[0]
            action = np.argmax(minibatch[self.targets].iloc[idx].tolist())  # self._hash_action(minibatch[self.targets].iloc[idx].tolist())
            done = np.isnan(next_state[0, 0])
            if not DDQN:
                # Vanilla DQN
                target = reward + self.gamma * np.max(self.model.predict(next_state)[0]) * np.invert(done)
            else:
                # Double DQN
                action_index = np.argmax(self.model.predict(next_state)[0])
                target = reward + self.gamma * self.target_model.predict(next_state)[0][action_index] * np.invert(done)

            targets_f[idx][action] = target
            # Filtering out states and targets for training

        history = self.model.fit(states, targets_f, epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def hash_to_action(x):
    return int(''.join(map(str, x)))

if __name__ == "__main__":
    # env = gym.make('CartPole-v1')
    targets = ['ANTIHYPERGLYCEMIC, BIGUANIDE TYPE',
                'INSULINS',
                'ANTIHYPERGLYCEMIC,INSULIN-RELEASE STIM.-BIGUANIDE',
                'ANTIHYPERGLYCEMIC, DPP-4 INHIBITORS',
                'ANTIHYPERGLY,INCRETIN MIMETIC(GLP-1 RECEP.AGONIST)',
                'ANTIHYPERGLYCEMIC, INSULIN-RELEASE STIMULANT TYPE',
                'ANTIHYPERGLYCEMIC,DPP-4 INHIBITOR-BIGUANIDE COMBS.',
                'ANTIHYPERGLYCEMC-SOD/GLUC COTRANSPORT2(SGLT2)INHIB',
                'ANTIHYPERGLYCEMIC, ALPHA-GLUCOSIDASE INHIBITORS',
                'ANTIHYPERGLYCEMIC, THIAZOLIDINEDIONE AND BIGUANIDE',
                'ANTIHYPERGLYCEMIC,THIAZOLIDINEDIONE(PPARG AGONIST)',
                'ANTIHYPERGLYCEMIC - DOPAMINE RECEPTOR AGONISTS',
                'ANTIHYPERGLYCEMIC-SGLT2 INHIBITOR-BIGUANIDE COMBS.',
                'ANTIHYPERGLYCEMIC, SGLT-2 AND DPP-4 INHIBITOR COMB',
                'ANTIHYPERGLYCEMIC, THIAZOLIDINEDIONE-SULFONYLUREA',
                'ANTIHYPERGLY,DPP-4 ENZYME INHIB.-THIAZOLIDINEDIONE',
                'ANTIHYPERGLY,INSULIN,LONG ACT-GLP-1 RECEPT.AGONIST',
                'ANTIHYPERGLYCEMIC, AMYLIN ANALOG-TYPE']
    targets = [ftr.lower().replace(' ','_') for ftr in targets]
    excluded = ['study_id', 'encounter_dt_ran']

    data_path = '~/Research/PHD/project/Hua Zheng/previous code/cleaned_EHR_treatment_param_lab_test_final_3diseases_cvd_encounter.csv'

    data = pd.read_csv(data_path)
    if diagnosis_reward:
        data['reward'] = data['reward_diagnosis']
        data.drop('reward_diagnosis', inplace = True)

    # data = data.drop(['egfr_mdrd_african_american_min', 'egfr_mdrd_african_american_max', 'egfr_mdrd_african_american', 'egfr_mdrd_non_african_american', 'egfr_mdrd_non_african_american_max','egfr_mdrd_non_african_american_min', 'next_egfr_mdrd_african_american','next_egfr_mdrd_african_american_max','next_egfr_mdrd_african_american_min', 'next_egfr_mdrd_non_african_american','next_egfr_mdrd_non_african_american_max', 'next_egfr_mdrd_non_african_american_min', 'bulk_chemicals_hist', 'next_bulk_chemicals_hist'],axis=1)
    data = data.drop(['egfr_mdrd_african_american_min', 'egfr_mdrd_african_american_max', 'egfr_mdrd_african_american', 'egfr_mdrd_non_african_american', 'egfr_mdrd_non_african_american_max','egfr_mdrd_non_african_american_min', 'next_egfr_mdrd_african_american','next_egfr_mdrd_african_american_max','next_egfr_mdrd_african_american_min', 'next_egfr_mdrd_non_african_american','next_egfr_mdrd_non_african_american_max', 'next_egfr_mdrd_non_african_american_min'],axis=1)
    data = data.dropna()
    treatment_cols = list(data.columns[116:159])
    label_cols = targets
    # remove irrelatvent cases
    data = data[data[label_cols].apply(lambda x: sum(x) > 0, axis=1)]

    target = data[label_cols].apply(lambda x: hash_to_action(x), axis=1)
    data['target'] = target
    counter = collections.Counter(target)

    size_of_actions = 20
    target_set = set([i[0] for i in counter.most_common(size_of_actions)])
    target_replacement = dict(zip(iter(target_set), range(size_of_actions)))
    target_column_renames = ['target' + str(i) for i in range(size_of_actions)]
    data = data[data.target.apply(lambda x: x in target_set)]
    data['target'] = data['target'].replace(target_replacement)
    reward_cols = ['reward_diabetes']

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

    state_cols = list(set(data.columns) - set(treatment_cols) - set(['target', 'CVD']) - set(['reward', 'reward_bp', 'reward_ascvd', 'reward_diabetes']) - set(['study_id', 'encounter_dt_ran']) - set(
        next_state_cols))
    #next_state_cols = ['next_' + s for s in list(state_cols)]
    patients_set = set(data['study_id'])
    data = data.drop(excluded, axis=1)

    normalized_df = (data.drop('target',axis=1) - data.drop('target',axis=1).min()) / (data.drop('target',axis=1).max() - data.drop('target',axis=1).min())
    normalized_df.fillna(1.0, inplace=True)
    ohe = to_categorical(data['target'], 100)
    for i, col in enumerate(target_column_renames):
        normalized_df[col] = ohe[:,i]

    train, test = train_test_split(normalized_df, test_size=0.4, random_state=2019)

    #x_train = train[state_cols]
    #y_train_class = train[label_cols]
    # y_train_category = train[['RX_CATEGORY']]

    #x_test = test[state_cols]
    #y_test_class = test[label_cols]

    state_size = len(state_cols)
    action_size = len(target_column_renames)
    agent = DQNAgent(state_size, action_size, target_column_renames, state_cols, reward_cols, next_state_cols)

    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 64
    cur_val_error = 0
    episodes_till_target_update = 100
    interested_train = train #train[train[targets].apply(lambda x: sum(x) > 0, axis=1)]
    interested_tests = test #test[test[targets].apply(lambda x: sum(x) > 0, axis=1)]

    for e in range(EPISODES):

        # patients_trajectories = random.sample(patients_set, batch_size)
        minibatch = interested_train.sample(batch_size)
        # minibatch = random.sample(agent.memory, batch_size)
        if e % episodes_till_target_update:
            agent.update_target()

        agent.learning_rate = agent.learning_rate * (1 - agent.learning_rate_decay)
        loss = agent.replay(minibatch, True)

        if e % 500 == 0:
            cur_val_error = sum(np.argmax(interested_tests[target_column_renames].values, axis=1) == np.argmax(
                agent.model.predict(interested_tests[state_cols]), axis=1)) / interested_tests.shape[0]
        # Logging training loss every 10 timesteps
        if e % 10 == 0:
            print("episode: {}/{}, loss: {:.5f}, test_acc: {:.4f}"
                .format(e, EPISODES, loss, cur_val_error))

    ''' Simulation based validation
    '''
    from sklearn.decomposition import PCA

    # pca = PCA(n_components=8)
    pca = PCA(.90)
    pca.fit(interested_tests[state_cols])
    principalComponents = pca.transform(interested_tests[state_cols])
    y = np.append(np.argmax(interested_tests[target_column_renames].values, axis=1).reshape([interested_tests.shape[0],1]), np.argmax(agent.model.predict(interested_tests[state_cols]), axis=1).reshape([interested_tests.shape[0],1]), axis=1)
    principalComponents = np.append(principalComponents[:, :23], y, axis=1)

    # kdt = KDTree(principalComponents, leaf_size=30, metric='euclidean')
    # similar_cases = kdt.query(principalComponents, k=10, return_distance=False)
    from sklearn.neighbors import NearestNeighbors, BallTree, KDTree

    # test = pd.read_csv("~/Research/PHD/project/Hua Zheng/previous code/cleaned_EHR_treatment_param_lab_test_final-test.csv")
    # principalComponents = pd.read_csv('~/Research/PHD/project/Hua Zheng/previous code/cleaned_EHR_treatment_param_lab_test_final-pca.csv', header = None)
    def dist(x, y):
        if x[-2] != y[-1] or x[-1] != y[-2]:
            return float('inf')

        return np.sqrt(np.sum(pow(x[:23]-y[:23], 2)))


    # matched_indices = principalComponents.iloc[:, -2] == principalComponents.iloc[:, -1]
    # unmatched_indices = principalComponents.iloc[:, -2] != principalComponents.iloc[:, -1]
    # principalComponents.loc[:,21:22] = principalComponents.loc[:, 21:22] * 1e8
    matched_indices = principalComponents[:, -2] == principalComponents[:, -1]
    unmatched_indices = principalComponents[:, -2] != principalComponents[:, -1]
    principalComponents[:, 23:25] = principalComponents[:, 23:25] * 1e8

    tree = BallTree(principalComponents[:, :24], leaf_size=1000, metric='euclidean')
    # nbrs = tree.query(principalComponents.values[unmatched_indices][:,list(range(21)) + [22]], k=10, return_distance=False)
    nbrs = tree.query(principalComponents[:,list(range(23)) + [24]], k=10, return_distance=False)
    nbrs_matched = tree.query(principalComponents[matched_indices][:, list(range(23)) + [24]], k=10, return_distance=False)
    # rearrange the index for interested tests
    interested_tests['next_hemoglobin_a1c_raw'] = interested_tests['next_hemoglobin_a1c'] * (data['next_hemoglobin_a1c'].max() - data['next_hemoglobin_a1c'].min()) + data['next_hemoglobin_a1c'].min()
    interested_tests.index = np.arange(0, len(interested_tests))
    test_bp_dict = interested_tests[['next_hemoglobin_a1c_raw']].to_dict('index')

    temp_df = interested_tests[matched_indices.tolist()][
        ['next_hemoglobin_a1c_raw']]
    temp_df.index = np.arange(0, len(temp_df))
    matched_test_bp_dict = temp_df.to_dict('index')

    # validate
    gain_hemoglobin_a1c = []
    for idx, v in enumerate(nbrs_matched):
        hemoglobin_a1c = .0
        for k in v:
            hemoglobin_a1c += test_bp_dict[k]['next_hemoglobin_a1c_raw']
        gain_hemoglobin_a1c.append(hemoglobin_a1c / 10.0 - matched_test_bp_dict[idx]['next_hemoglobin_a1c_raw'])

    print(np.mean(gain_hemoglobin_a1c))

    gain_hemoglobin_a1c = []
    for idx, v in enumerate(nbrs):
        hemoglobin_a1c = .0
        for k in v:
            hemoglobin_a1c += test_bp_dict[k]['next_hemoglobin_a1c_raw']
        gain_hemoglobin_a1c.append(hemoglobin_a1c / 10.0 - test_bp_dict[idx]['next_hemoglobin_a1c_raw'])

    print(np.mean(gain_hemoglobin_a1c))
    # if e % 10 == 0:
    #     agent.save("./save/cartpole-dqn.h5")

    plt.legend(prop={'size': 12})
    plt.title('Treatment Chart Doctor vs RL')
    plt.xlabel('Therapy line')
    plt.ylabel('Frequency')
    temp_principalComponents = pd.DataFrame(principalComponents[:, 23:25]/1e8)
    temp_principalComponents.columns = ['Doctor', 'RL']
    # sns.countplot(x="Doctor", hue="RL", data=temp_principalComponents)
    # sns.countplot(x="RL", hue="Doctor", data=temp_principalComponents)
    # temp_principalComponents.to_csv("~/Research/PHD/project/Hua Zheng/previous code/temp_principalComponents.csv", index=False)
    # temp_principalComponents= pd.read_csv("~/Research/PHD/project/Hua Zheng/previous code/temp_principalComponents.csv")
    sns.distplot(temp_principalComponents['Doctor'], kde=False, label='Doctor & RL', hist=True)
    sns.distplot(temp_principalComponents['RL'], kde=False, label='Doctor', hist=True)

    plt.title('Treatment Chart Doctor vs RL')
    plt.xlabel('Therapy line')
    plt.ylabel('Frequency')

    plt.show()

    agent.model = load_model('model-5rounds-l3-tanh-lr0004.h5')
    x_train = train[state_cols]
    x_test = test[state_cols]

    background = x_train.values[np.random.choice(x_train.shape[0], 100, replace=False)]
    e = shap.DeepExplainer(agent.model, background)
    shap_values = e.shap_values(x_test.values[:10])
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    agent.model(test[targets])
