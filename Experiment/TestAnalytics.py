import Utils

ln = ["Idea2_zeros","Idea2_ones","Baseline_myopic","Baseline_farsighted","Bound1_myopic","Bound1_farsighted","Thompson_baseline_myopic","Thompson_baseline_farsighted","BayesUCBPersistentfarsighted","BayesUCBPersistentmyopic"]

#ln = ["Idea2_zeros","Idea2_ones","Baseline_myopic","Bound1_myopic","Thompson_baseline","BayesUCBPersistent"]

avg_list = []
std_list = []
ids = [0,1,2,3,4,5]

tmax_list = ["210"]
for x in tmax_list:
    Utils.compute_experiment_pulled_arms(n_runs=20,time_horizon=20000,ids=ids,learner_names=ln,experiment_name="experiment_C_"+x)

'''
Utils.compute_full_analytics_from_files(n_runs=20,time_horizon=20000,Oracle_best_arm_value=10.823308799999998,learner_names=ln,experiment_name="experiment_C_5")
Utils.compute_full_analytics_from_files(n_runs=20,time_horizon=20000,Oracle_best_arm_value=19.166313650000003,learner_names=ln,experiment_name="experiment_C_10")
Utils.compute_full_analytics_from_files(n_runs=20,time_horizon=20000,Oracle_best_arm_value=36.28571441774977,learner_names=ln,experiment_name="experiment_C_20")
Utils.compute_full_analytics_from_files(n_runs=20,time_horizon=20000,Oracle_best_arm_value=70.57142857247196,learner_names=ln,experiment_name="experiment_C_40")
Utils.compute_full_analytics_from_files(n_runs=20,time_horizon=20000,Oracle_best_arm_value=122.00000000002078,learner_names=ln,experiment_name="experiment_C_70")
Utils.compute_full_analytics_from_files(n_runs=20,time_horizon=20000,Oracle_best_arm_value=190.57142857142938,learner_names=ln,experiment_name="experiment_C_110")
Utils.compute_full_analytics_from_files(n_runs=20,time_horizon=20000,Oracle_best_arm_value=276.28571428571433,learner_names=ln,experiment_name="experiment_C_160")
Utils.compute_full_analytics_from_files(n_runs=20,time_horizon=20000,Oracle_best_arm_value=362.0,learner_names=ln,experiment_name="experiment_C_210")
'''