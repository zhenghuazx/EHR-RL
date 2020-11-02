# EHR-RL
Repository for RL based prescriptive algorithm for the paper 
"Zheng, H., et al., In Press. Personalized multimorbidity management for patients with type 2 diabetes using reinforcement learning of electronic health records. Drugs."

The research was conducted by Hua Zheng and supervised by Professor Wei Xie, Professor Judy Zhong aand Professor Ilya O. Ryzhov. Results and publications are primarily coming out from Professor Wei Xie's research group. The paper has been accepted by Drugs and the preprint can be found in [paper](http://www1.coe.neu.edu/~wxie/RL_EHR_paper-2020.pdf). We would appreciate a citation if you use the code or results!


Outline of our study:

# AIMS: Comorbid chronic conditions are common among people with type 2 diabetes. We developed an Artificial Intelligence algorithm, based on Reinforcement Learning (RL), for personalized diabetes and multimorbidity management with strong potential to improve health outcomes relative to current clinical practice.

# METHODS: We modeled glycemia, blood pressure and cardiovascular disease (CVD) risk as health outcomes using a retrospective cohort of 16,665 patients with type 2 diabetes from New York University Langone Health ambulatory care electronic health records in 2009 to 2017. We trained a RL prescription algorithm that recommends a treatment regimen optimizing patients’ cumulative health outcomes using their individual characteristics and medical history at each encounter. The RL recommendations were evaluated on an independent subset of patients. 

# RESULTS: The single-outcome optimization RL algorithms, RL-glycemia, RL-blood pressure, and RL-CVD, recommended consistent prescriptions with what observed by clinicians in 86.1%, 82.9% and 98.4% of the encounters. For patient encounters in which the RL recommendations differed from the clinician prescriptions, significantly fewer encounters showed uncontrolled glycemia (A1c>8% on 35% of encounters), uncontrolled hypertension (blood pressure > 140mmHg on 16% of encounters) and high CVD risk (risk > 20% on 25% of encounters) under RL algorithms than those observed under clinicians (43%, 27% and 31% of encounters respectively; all P < 0.001).  

# CONCLUSIONS: A personalized reinforcement learning prescriptive framework for type 2 diabetes yielded high concordance with clinicians’ prescriptions and substantial improvements in glycemia, blood pressure, cardiovascular disease risk outcomes.
