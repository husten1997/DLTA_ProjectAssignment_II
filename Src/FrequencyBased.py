#%% Freq Vec
from tqdm import tqdm

term_list = data['proText_noStop'].str.cat().split()
term_list = np.unique(term_list)

#%%
freq_vec = {}
for w in tqdm(term_list):
    freq_vec[w] = data['proText_noStop'].str.cat().split().count(w)

freq_vec = pd.Series(freq_vec)
freq_vec = freq_vec.sort_values()


#%% Freq Matrix
freq_matrix = pd.DataFrame(columns=term_list)

for i in tqdm(range(data.shape[0])):
    freq_matrix.loc[i, :] = [data.loc[i, 'proText_noStop'].split().count(w) for w in term_list]
