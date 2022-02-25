from label_studio_sdk import Client
from label_studio_sdk import project
from label_studio_sdk import project
import numpy as np
import pandas as pd
from os.path import exists

# Function for Data Download and Preparation
def DataDownload():
    #%%
    LABEL_STUDIO_URL = 'http://132.231.59.226:8080'  # this address needs to be the same as the address the label-studio is hosted on.
    API_KEY = '430d989b7f9723b1e5d82462ded8a22bfb331c6d'  # please add your personal API_Key here to get your API_Key follow the Pictures below

    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()
    pro = project.Project.get_from_id(ls, "1")
    tasks = project.Project.get_labeled_tasks(pro)

    # %%
    data_raw = {}
    data_raw['start'] = []
    data_raw['end'] = []
    data_raw['text'] = []
    data_raw['label'] = []

    for e in tasks[0]['annotations'][0]['result']:
        # print(e['value']['start'])
        data_raw['start'].append(e['value']['start'])
        data_raw['end'].append(e['value']['end'])
        data_raw['text'].append(e['value']['text'])
        data_raw['label'].append(e['value']['labels'][0])

    data = pd.DataFrame(data_raw)
    data['id'] = list(range(0, data.shape[0]))

    # %%
    start_vec = np.array(data['start']).reshape((-1, 1))
    helper = np.ones(len(start_vec)).reshape((1, -1))

    A = np.matmul(start_vec, helper)

    dist = A - A.T

    # %%
    uncert = 10
    grouping_vec = list()
    for i in range(0, dist.shape[0]):
        helper = list()
        for j in range(0, dist.shape[0]):
            if abs(dist[i][j]) < uncert: helper.append(j)
        grouping_vec.append(pd.Series(helper))

    grouping_vec = [list(x) for x in set(tuple(x) for x in grouping_vec)]
    # grouping_vec = pd.unique(pd.Series(grouping_vec))

    # %%
    data_all_raw = {}
    data_all_raw['start'] = []
    data_all_raw['end'] = []
    data_all_raw['text'] = []
    data_all_raw['label_id'] = []
    data_all_raw['label_l1'] = []
    data_all_raw['label_l2'] = []
    data_all_raw['label_l3'] = []

    mylen = np.vectorize(len)

    def arrayContains(x, y):
        output = []
        for a in x:
            output.append(str(a).__contains__(y))

        return output

    for e in grouping_vec:
        helper = data.loc[e, :]
        print(helper)
        data_all_raw['start'].append(helper['start'].min())
        data_all_raw['end'].append(helper['start'].max())
        data_all_raw['text'].append(
            str(list(helper.loc[mylen(helper['text']) == mylen(helper['text']).max(), 'text'])[0]))
        data_all_raw['label_id'].append(str(list(helper.loc[arrayContains(helper['label'], "ID"), 'label'])[0]))
        data_all_raw['label_l1'].append(str(list(helper.loc[arrayContains(helper['label'], "_1_"), 'label'])[0]))
        data_all_raw['label_l2'].append(str(list(helper.loc[arrayContains(helper['label'], "_2_"), 'label'])[0]))
        data_all_raw['label_l3'].append(str(list(helper.loc[arrayContains(helper['label'], "_3_"), 'label'])[0]))

    data_all = pd.DataFrame(data_all_raw)

    # %%
    def arrayContains(x, y):
        output = []
        for a in x:
            output.append(str(a).__contains__(y))

        return output

    strange_list = pd.DataFrame(columns=['docID', 'start', 'end', 'text', 'label', 'id'])

    def selectLabel(array, label, first=True):
        label_array = list(array.loc[arrayContains(array['label'], label), 'label'])
        if len(label_array) == 1:
            return str(label_array[0])
        elif len(label_array) == 0:
            globals()['strange_list'] = globals()['strange_list'].append(pd.DataFrame(array), ignore_index=True)
            return str(np.NAN)
        else:
            globals()['strange_list'] = globals()['strange_list'].append(pd.DataFrame(array), ignore_index=True)
            if len(np.unique(label_array)) == 1:
                return str(np.unique(label_array)[0])
            elif first:
                return str(label_array[0])
            else:
                return str(label_array)

    mylen = np.vectorize(len)

    data_all_raw = {}
    data_all_raw['docID'] = []
    data_all_raw['start'] = []
    data_all_raw['end'] = []
    data_all_raw['text'] = []
    data_all_raw['label_id'] = []
    data_all_raw['label_l1'] = []
    data_all_raw['label_l2'] = []
    data_all_raw['label_l3'] = []

    for t in range(0, len(tasks)):
        print(t)
        # Acquire Raw Data
        data_raw = {}
        data_raw['docID'] = []
        data_raw['start'] = []
        data_raw['end'] = []
        data_raw['text'] = []
        data_raw['label'] = []

        for e in tasks[t]['annotations'][0]['result']:
            data_raw['docID'].append(tasks[t]['id'])
            data_raw['start'].append(e['value']['start'])
            data_raw['end'].append(e['value']['end'])
            data_raw['text'].append(e['value']['text'])
            data_raw['label'].append(e['value']['labels'][0])

        data = pd.DataFrame(data_raw)
        data['id'] = list(range(0, data.shape[0]))

        # Create Dis Matrix
        start_vec = np.array(data['start']).reshape((-1, 1))
        helper = np.ones(len(start_vec)).reshape((1, -1))

        A = np.matmul(start_vec, helper)

        dist = A - A.T

        # Create Groups which are most similar to each other
        uncert = 10
        grouping_vec = list()
        for i in range(0, dist.shape[0]):
            helper = list()
            for j in range(0, dist.shape[0]):
                if abs(dist[i][j]) < uncert: helper.append(j)
            grouping_vec.append(helper)

        grouping_vec = [list(x) for x in set(tuple(x) for x in grouping_vec)]

        # Create dic with the values
        for e in grouping_vec:
            helper = data.loc[e, :]
            # print(helper)
            data_all_raw['docID'].append(tasks[t]['id'])
            data_all_raw['start'].append(helper['start'].min())
            data_all_raw['end'].append(helper['start'].max())
            data_all_raw['text'].append(
                str(list(helper.loc[mylen(helper['text']) == mylen(helper['text']).max(), 'text'])[0]))
            data_all_raw['label_id'].append(selectLabel(helper, "ID"))
            data_all_raw['label_l1'].append(selectLabel(helper, "_1_"))
            data_all_raw['label_l2'].append(selectLabel(helper, "_2_"))
            data_all_raw['label_l3'].append(selectLabel(helper, "_3_"))

    data_all = pd.DataFrame(data_all_raw)
    strange_list = pd.DataFrame(strange_list)

    data_all.to_csv("Data/all_data.csv")
    data_all.to_excel("Data/all_data.xlsx")
    strange_list.to_csv("Data/strange_list.csv")
    strange_list.to_excel("Data/strange_list.xlsx")

# Function for data import
def DataImport():
    if not exists("Data/all_data.csv"):
        DataDownload()

    data = pd.read_csv("Data/all_data.csv")
    return data

