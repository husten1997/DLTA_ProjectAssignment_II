from label_studio_sdk import Client
from label_studio_sdk import project
from label_studio_sdk import project
import numpy as np
import pandas as pd
from os.path import exists


# Wrapper function for data download and import. It checks if a *.csv file is present and handles the data download and
# data transformation if it's not.
#
#   overwrite: Boolean value which determines if the available csv files would be overwritten by new ones
#              (means DataDownload is used, even though the data is locally available)
def DataImport(overwrite = False):
    if not exists("Data/all_data.csv") or overwrite:
        DataDownload()

    data = pd.read_csv("Data/all_data.csv", index_col = 0)
    return data


# Function which covers the data download and transformation. Therefore the raw data is fetched from the server
# (requires VPN). Afterwards two helper functions are created, one which basically an element-wise application of
# str.contains to a list. The other helper function contains the logic required for the label selection if more
# (or less) than 4 labels are available for one text (=Question/Answer). Then a dictionary for the raw_data is created,
# which will contain a list for "strat"-, "end"-, "text"- and "label"-data. Then a for loop iterates through the raw
# data and appends the information to the respective lists.
# This results in a DataFrame which contains an entry for each label, or in other words for each question/answer in
# each doc (ideally) four entries with ID, and the three question/answer levels.
# Next step shall be combining the 4 stages of the labels together. This requires a matching of the text and/or
# start/end information. Because the text and/or start/end information for each label can not be directly matched
# (because sometimes there are additional characters included at the beginning or end of the text, therefore the stat
# and end information can also be different) a matrix with the absolute distances is calculated.
# Afterwards an uncertainty factor is set (to a distance of 10) and a list of all elements which have a distance of less
# than 10 to each other is created. This ensures that the entries which are the closest to each other are
# combined/matched. Now the label information of each stage can be combined. For that a for loop iterates through each
# "label-group" ("label-group" refers to the 3, 4 or 5 labels which are closest to each other) and selects the
# label-name to one of the for label-stages (QID/AID, label_l1, label_l2, label_l3). If more than 1 label is present
# for one label stage (for example if a question has two times question_3_* labels) in a first step the algorithm
# tries to use np.unique to see if the two question_3_* labels are actually different. If thats the case, one can just
# use one of the two labels without loss of information. If that's NOT the case, one has to choose to drop information
# (then only the FIRST of the two labels is used) or include the information (then an array of the two labels is used,
# which obviously breaks the unique labeling if each text).
# After four label stages are matched to the text, one obtains a DataFrame with one column for text and four columns
# for the label stages.
# This DataFrame is then saved as csv and can be loaded at a later point in time.
def DataDownload():
    #%%
    LABEL_STUDIO_URL = 'http://132.231.59.226:8080'  # this address needs to be the same as the address the label-studio is hosted on.
    API_KEY = '430d989b7f9723b1e5d82462ded8a22bfb331c6d'  # please add your personal API_Key here to get your API_Key follow the Pictures below

    ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
    ls.check_connection()
    pro = project.Project.get_from_id(ls, "1")
    tasks = project.Project.get_labeled_tasks(pro)

    # %%
    def arrayContains(x, y):
        output = []
        for a in x:
            output.append(str(a).__contains__(y))

        return output

    strange_list = pd.DataFrame(columns=['docID', 'start', 'end', 'text', 'label', 'id'])

    def selectLabel(array, label, first=True):
        nonlocal strange_list
        label_array = list(array.loc[arrayContains(array['label'], label), 'label'])
        if len(label_array) == 1:
            return str(label_array[0])
        elif len(label_array) == 0:
            strange_list = strange_list.append(pd.DataFrame(array), ignore_index=True)
            return str(np.NAN)
        else:
            strange_list = strange_list.append(pd.DataFrame(array), ignore_index=True)
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
        #print(t)
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




