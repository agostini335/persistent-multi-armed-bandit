#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from datetime import datetime
import matplotlib.lines as mlines
from tqdm import tqdm
import hashlib

#LOADING
s_df_a = pd.read_csv("/home/ago/Documenti/DATASET_SPOTIFY/log_0_20180715_000000000000.csv",sep=",")
s_df_b = pd.read_csv("/home/ago/Documenti/DATASET_SPOTIFY/log_0_20180716_000000000000.csv",sep=",")
s_df_c = pd.read_csv("/home/ago/Documenti/DATASET_SPOTIFY/log_0_20180717_000000000000.csv",sep=",")
s_df_d = pd.read_csv("/home/ago/Documenti/DATASET_SPOTIFY/log_0_20180818_000000000000.csv",sep=",")
s_df_e = pd.read_csv("/home/ago/Documenti/DATASET_SPOTIFY/log_0_20180819_000000000000.csv",sep=",")
s_df_f = pd.read_csv("/home/ago/Documenti/DATASET_SPOTIFY/log_0_20180820_000000000000.csv",sep=",")
s_df_g = pd.read_csv("/home/ago/Documenti/DATASET_SPOTIFY/log_0_20180821_000000000000.csv",sep=",")
#%%PREPROCESSING
s_df = s_df_a.append(s_df_b,ignore_index=True)
s_df = s_df.append(s_df_c,ignore_index=True)
s_df = s_df.append(s_df_d,ignore_index=True)
s_df = s_df.append(s_df_e,ignore_index=True)
s_df = s_df.append(s_df_f,ignore_index=True)
s_df = s_df.append(s_df_g,ignore_index=True)
s_df = s_df[s_df['context_type'] == 'editorial_playlist']
s_df = s_df[s_df['session_length'] == 20]
s_df = s_df[s_df['context_switch'] == 0]
s_df['counts'] = s_df.groupby(['session_id'])['session_id'].transform('count')
s_df = s_df[s_df['counts']==20]
s_df.sort_values(by=['session_id','track_id_clean'], inplace=True)


print(s_df.head())
print(len(s_df))

#%%HASH
dict_playlist_ids = dict()
grouped = s_df.groupby('session_id')
for name, group in grouped:
    #dict_playlist_ids[name] = hash("".join(group['track_id_clean'].values))
    dict_playlist_ids[name] = ("".join(group['track_id_clean'].values))

def get_playlist_id(x):
    return dict_playlist_ids[x]

s_df['playlist_id'] = s_df['session_id'].apply(lambda x :  get_playlist_id(x))

#%% SESSION COUNT
dict_playlist_count = dict()
x = s_df.groupby('playlist_id').session_id.nunique().sort_values(ascending = False)
x = x.head(30)
print(x)
s_df["session_count"] = s_df.groupby(['playlist_id'])['session_id'].transform('nunique')
s_df.sort_values(by = ['session_count','playlist_id'],ascending=False,inplace=True)

#%%CUT
s_df_old = s_df
s_df = s_df[s_df['session_count']>10]
#s_df = s_df[s_df['session_count']<1000]
#print(s_df.head())

#%%
#s_df = s_df_old
#s_df = s_df[s_df['session_count']>40]



#%%PLAYLIST->SONG
playlist_song_dict = dict()
playlist_unique_ids = list(s_df["playlist_id"].unique())
total_song_selected = []

for playlist_id in list(playlist_unique_ids):
    song_list = list( s_df[s_df["playlist_id"]==playlist_id]["track_id_clean"].unique())
    if(len(song_list)<20):
        print(len(song_list))
        #remove less than 20 songs
        s_df = s_df[s_df["playlist_id"]!=playlist_id]
        playlist_unique_ids.remove(playlist_id)
    else:
        assert(len(song_list)==20)
        playlist_song_dict[playlist_id] = song_list
        total_song_selected = total_song_selected + song_list

total_song_selected = list(total_song_selected)
total_song_sel_set = set(total_song_selected)

occurence = []
for s in total_song_sel_set:
    count = 0
    for playlist_id in playlist_unique_ids:
        count += playlist_song_dict[playlist_id].count(s)
    occurence.append(count)

occurence.sort()
#print(occurence)
print(len(total_song_sel_set))
print(len(total_song_selected))
#%% PLAYLIST SIMILARITY
def sim(p1_id,p2_id):
    s1 = set(playlist_song_dict[p1])
    s2 = set(playlist_song_dict[p2])
    return len(s1.intersection(s2))



p_simil_dict = dict()
for p1 in playlist_unique_ids:
    p_simil_dict[p1] = []
    for p2 in playlist_unique_ids:
        p_simil_dict[p1].append(sim(p1,p2))

print(p_simil_dict)

#%% PLAYLIST SELECTION SEQUENTIAL
'''
import random
best_config = []
temp_best = []
for i in range(10000):
    random.shuffle(playlist_unique_ids)
    selected_playlist_ids = []
    for p in playlist_unique_ids:
        flag_add = True
        
        for sp in selected_playlist_ids:
            s1 = set(playlist_song_dict[p])
            s2 = set(playlist_song_dict[sp])
            if (len(s1.intersection(s2))!=0):
                flag_add = False
                break
        
        if flag_add:
            selected_playlist_ids.append(p)

    if (len(selected_playlist_ids)>7 and set(temp_best).intersection(set(selected_playlist_ids))!=0):
        best_config.append(selected_playlist_ids)
        temp_best =selected_playlist_ids
print(best_config)
'''


selected_playlist_ids = []
while(len(selected_playlist_ids)<20):
        for p in playlist_unique_ids:
                flag_add = True
                for sp in selected_playlist_ids:
                        s1 = set(playlist_song_dict[p])
                        s2 = set(playlist_song_dict[sp])
                        if (len(s1.intersection(s2))!=0):
                                flag_add = False
                                break
                        
                if flag_add:
                        selected_playlist_ids.append(p)

 
print(selected_playlist_ids)
#TOP SELECTED




#%% PLAYLIST SELECTION SEQUENTIAL...
'''
[-3598320077111359846, 8983709499987089383, 7646713046563469981, -6676014647941283664, 6191756317457639284, -4801074207345966596, 1894549564329788058, -4921676648624111352, 7635003863785592365, 7230074010306764606, -9118883917082150466, 2497448364807303530, 958254126374285448, 6693239639148122170, 8356955581625804858, -188137721825155330, 8530666716719734727, 4038455966363844056, 6621244533133513708, 2877742255107575377, 7094408320320362863

'''
#%%


#%%
