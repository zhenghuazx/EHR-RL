# EHR-RL
Repository for RL based prescriptive algorithm for the paper 
"Zheng, H., et al., In Press. Personalized multimorbidity management for patients with type 2 diabetes using reinforcement learning of electronic health records. Drugs."

The research was conducted by Hua Zheng and supervised by Professor Wei Xie, Professor Judy Zhong aand Professor Ilya O. Ryzhov. Results and publications are primarily coming out from Professor Wei Xie's research group. The paper has been accepted by Drugs and the preprint can be found in [paper](http://www1.coe.neu.edu/~wxie/RL_EHR_paper-2020.pdf). We would appreciate a citation if you use the code or results!


# Outline of our study:

* AIMS: Comorbid chronic conditions are common among people with type 2 diabetes. We developed an Artificial Intelligence algorithm, based on Reinforcement Learning (RL), for personalized diabetes and multimorbidity management with strong potential to improve health outcomes relative to current clinical practice.

* METHODS: We modeled glycemia, blood pressure and cardiovascular disease (CVD) risk as health outcomes using a retrospective cohort of 16,665 patients with type 2 diabetes from New York University Langone Health ambulatory care electronic health records in 2009 to 2017. We trained a RL prescription algorithm that recommends a treatment regimen optimizing patients’ cumulative health outcomes using their individual characteristics and medical history at each encounter. The RL recommendations were evaluated on an independent subset of patients. 

* RESULTS: The single-outcome optimization RL algorithms, RL-glycemia, RL-blood pressure, and RL-CVD, recommended consistent prescriptions with what observed by clinicians in 86.1%, 82.9% and 98.4% of the encounters. For patient encounters in which the RL recommendations differed from the clinician prescriptions, significantly fewer encounters showed uncontrolled glycemia (A1c>8% on 35% of encounters), uncontrolled hypertension (blood pressure > 140mmHg on 16% of encounters) and high CVD risk (risk > 20% on 25% of encounters) under RL algorithms than those observed under clinicians (43%, 27% and 31% of encounters respectively; all P < 0.001).  

* CONCLUSIONS: A personalized reinforcement learning prescriptive framework for type 2 diabetes yielded high concordance with clinicians’ prescriptions and substantial improvements in glycemia, blood pressure, cardiovascular disease risk outcomes.

# Main Results
## Higher Efficacy
RL based prescriptions significantly improves patients' health outcomes and reduce the number of patients in serious conditions, i.e. SBP > 140 mmHg HbA1c > 8% and FHS CVD risk > 20%.
| __RL-glycemia__                                                                                        |                   |                          |         |
|--------------------------------------------------------------------------------------------------------|-------------------|--------------------------|---------|
| Encounters for which algorithm’s recommendation differed from observed Clinician's prescription (N(%)) | 15,578 (13.9)     |                          |         |
|                                                                                                        | RL-glycemia       | Clinician's prescription | P-value |
| A1c (Mean(SE))                                                                                         | 7.80 (0.01)       | 8.09 (0.01)              | <0.001  |
| A1c > 8% (N(%))                                                                                        | 5,421 (34.8)      | 6,617 (42.5)             | <0.001  |
| <div align="center"> __RL-Blood Pressure__                                                             |                   |                          |         |
| Encounters for which algorithm’s recommendation differed from observed Clinician's prescription (N(%)) | 20,251 (17.1)     |                          |         |
|                                                                                                        | RL-BP             | Clinician's prescription | P-value |
| SBP(Mean(SE))                                                                                          | 131.77(0.06)      | 132.35 (0.11)            | <0.001  |
| SBP > 140 mmHg (N(%))                                                                                  | 3,256 (16.1)      | 5,390 (26.6)             | <0.001  |
| RL-CVD                                                                                                 |                   |                          |         |
| Encounters for which algorithm’s recommendation differed from observed Clinician's prescription (N(%)) | 946 (1.6)         |                          |         |
|                                                                                                        | RL-CVD            | Clinician's prescription | P-value |
| FHS (Mean(SE))                                                                                         | 13.65 (0.26)      | 17.18 (0.36)             | <0.001  |
| FHS > 20% (N(%))                                                                                       | 237 (25.1)        | 299 (31.6)               | <0.001  |
| <div align="center"> __RL-multimorbidity__                                                             |                   |                          |         |
| Encounters for which algorithm’s recommendation differed from observed Clinician's prescription (N(%)) | 102,184 (28.9)    |                          |         |
|                                                                                                        | RL-multimorbidity | Clinician's prescription | P-value |
| A1c (Mean(SE))                                                                                         | 7.14 (0.003)      | 7.19 (0.005)             | <0.001  |
| A1c > 8% (N(%))                                                                                        | 16,436 (16.08)    | 20,879 (20.43)           | <0.001  |
| SBP (Mean(SE))                                                                                         | 129.40 (0.03)     | 129.58 (0.05)            | <0.001  |
| SBP > 140 mmHg (N(%))                                                                                  | 9,800 (9.59)      | 20,957 (20.51)           | <0.001  |
| FHS (Mean(SE))                                                                                         | 21.89 (0.04)      | 25.61 (0.05)             | <0.001  |
| FHS > 20% (N(%))                                                                                       | 48,283 (47.3)     | 55,957 (54.8)            | <0.001  |

## Less prescriptions
Both AI and doctor has similar prescriotive distribution but AI tends to prescribe less than doctors:
![Prescriptive Distribution](https://github.com/zhenghuazx/EHR-RL/blob/master/distribution3.svg)

## Consistently performance acrosss different groups

| Subgroup   | Number of encounters | RL benefit relative to clinician policy (standard of care) |              |               |                   |                 |                 |               |
|------------|----------------------|------------------------------------------------------------|--------------|---------------|-------------------|-----------------|-----------------|---------------|
|            |                      | A1c                                                        | Systolic BP  | Triglycerides | Total Cholesterol | LDL Cholesterol | HDL Cholesterol | CVD Risk      |
| Male       | 43816                | -0.09 (0.01)                                               | -0.32 (0.07) | -5.27 (0.50)  | -0.10 (0.17)      | 0.08 (0.14)     | 0.71 (0.06)     | -5.09 (0.09)  |
| Female     | 58368                | -0.02 (0.01)                                               | -0.07 (0.06) | -1.99 (0.33)  | -1.23 (0.16)      | -0.61 (0.14)    | -0.41 (0.06)    | -2.68 (0.05)  |
| Age > 60   | 75924                | 0.01 (0.00)                                                | -0.59 (0.05) | -0.43 (0.29)  | -0.05 (0.13)      | 0.27 (0.12)     | -0.24 (0.05)    | -5.70 (0.06)  |
| Age ≤ 60   | 26260                | -0.23 (0.01)                                               | 1.02 (0.09)  | -11.97 (0.72) | -2.75 (0.25)      | -1.99 (0.21)    | 0.98 (0.08)     | 2.03 (0.07)   |
| White      | 60029                | -0.02 (0.00)                                               | -0.02 (0.06) | -3.78 (0.37)  | -1.60 (0.15)      | -1.12 (0.13)    | 0.16 (0.06)     | -4.04 (0.07)  |
| Black      | 31775                | -0.12 (0.01)                                               | -0.92 (0.09) | 1.79 (0.47)   | -0.52 (0.22)      | -0.43 (0.19)    | -0.64 (0.08)    | -3.39 (0.08)  |
| Other Race | 10380                | -0.02 (0.01)                                               | 1.17 (0.15)  | -17.02 (1.08) | 3.50 (0.39)       | 4.70 (0.33)     | 1.78 (0.13)     | -2.82 (0.14)  |
| Smoke      | 5747                 | -0.10 (0.02)                                               | -0.52 (0.20) | -16.54 (1.71) | -1.31 (0.55)      | -0.73 (0.46)    | 2.14 (0.17)     | -10.41 (0.26) |
| Non-Smoke  | 96437                | -0.05 (0.00)                                               | -0.16 (0.05) | -2.61 (0.28)  | -0.71 (0.12)      | -0.29 (0.10)    | -0.05 (0.05)    | -3.31 (0.05)  |
