# config.py

embedding_dim = 8
ts_hidden_dim = 32
tab_hidden_dim = 32
ts_output_dim = 64
tab_output_dim = 32
total_dim = ts_output_dim + tab_output_dim

ts_day = 3
batch = 64
projection_dim = 16
joint_hidden_dim = 32

tab_data_path = '../data/lifestyle_train.csv'
ts_data_path = '../data/lifelog_train.csv'