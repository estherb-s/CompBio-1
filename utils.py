import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn import preprocessing
import seaborn as sns
from os.path import exists
from download import download_tsv

def load_and_fillna(path):
    """
    Load data from csv to dataframe and fill missing with mean values

    Arguments:
        path {str} -- [path to csv file]

    Returns:
        [pandas.DataFrame] -- [dataframe loaded with CSV]
    """
    if not exists(path):
        print("That directory / file doesn't exist")
        link = str(input("Link to the zipped file: "))
        name = str(input("Filename to save as, include extensions such as .tsv: "))
        download_tsv(link=link, name=name)
        path = "data/"+name

    df = pd.read_csv(path, sep="\t", header=0)
    df.fillna(df.mean())
    df['id2'] = df.index
    df.set_index('Ensembl_ID',inplace=True)
    return df.T


def pca_prep(df, n_comp):
    """
    Perform PCA on the data to find the top N components

    Arguments:
        df {pandas.DataFrame} -- [dataframe loaded with CSV]
        n_comp {int} -- [number of top components to extract]

    Returns:
        [pandas.DataFrame] -- [returns a dataframe with top N components named {PC-*N*}]
    """
    df = df.iloc[1:,:]
    dataScaled = pd.DataFrame(preprocessing.scale(df),columns = df.columns)
    pca = PCA(n_components=n_comp)
    pca.fit(dataScaled)
    dfPCA = pca.transform(dataScaled)
    temp = ["PC-"+ str(x) for x in range(1,n_comp+1)]
    extended = pd.DataFrame(pca.components_,columns=dataScaled.columns,index = temp)
    return extended


def add_target(df, t):
    """
    Add target column to the dataframe

    Arguments:
        df {pandas.DataFrame} -- [DataFrame to be modified]
        t {int} -- [Target variable]

    Returns:
        [pandas.DataFrame] -- [modified DataFrame with the target]
    """

    df = df.T
    df['Target'] = t
    return df



def run_random_forest(df, random_state, n_estimators, n_importance,  debug=True, name=""):
    """
    Runs random forest on the data provided

    Arguments:
        df {pandas.DataFrame} -- DataFrame to be modified
        random_state {int} -- random state param for random forest
        n_estimators {int} -- number of estimators
        n_importance {int} -- Top N important features to extract

    Keyword Arguments:
        debug {bool} -- see print output logs if debug is true (default: {True})
        name {str} -- name to identify the chart better when running multiple random forests (default: {""})

    Returns:
        list -- top important respective features
    """
    features = list(df.columns[:-1])

    Y = df['Target']
    X = df[features]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2, random_state = random_state)


    # for debugging purposes
    if debug:
        print('Training Features Shape:', Xtrain.shape)
        print('Training Labels Shape:', Ytrain.shape)
        print('Testing Features Shape:', Xtest.shape)
        print('Testing Labels Shape:', Ytest.shape)

        print(Ytrain.index)

    randForest = RandomForestClassifier(n_estimators = n_estimators, random_state = random_state)

    print("Training the model")
    randForest.fit(Xtrain,Ytrain)

    f_importance = randForest.feature_importances_
    sortedIdx = np.argsort(f_importance)[-n_importance:]
    important_features = [features[i] for i in sortedIdx]
    plt.title(f'Random Forest Feature Importances for {name} data')
    plt.barh(range(len(sortedIdx)), f_importance[sortedIdx], color='b', align='center')
    plt.yticks(range(len(sortedIdx)), important_features)
    plt.xlabel('Relative Importance')
    plt.show()

    if debug:
        print(f'Top {n_importance} features: {important_features[::-1]}')

    return important_features[::-1]



def test():
    df = load_and_fillna("data/aa.tsv")
    df2 = load_and_fillna("data/TCGA-LUSC.tsv")
    principalCompLUAD = pca_prep(df, 5)
    principalCompLUSC = pca_prep(df2, 5)
    principalCompLUAD['Target'] = '0'
    principalCompLUSC['Target'] = '1'
    # clean up memory
    del df
    del df2
    lung = pd.concat([principalCompLUAD,principalCompLUSC])
    important = run_random_forest(df=lung, random_state=42, n_estimators=1000, n_importance=20, name="Lung")

if __name__ == "__main__":
    test()
