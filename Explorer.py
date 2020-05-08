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
    return extended.T


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



def run_random_forest(df, random_state, n_estimators, n_importance,  debug=True):
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
    plt.title('Random Forest Feature Importances')
    plt.barh(range(len(sortedIdx)), f_importance[sortedIdx], color='b', align='center')
    plt.yticks(range(len(sortedIdx)), [features[i] for i in sortedIdx])
    plt.xlabel('Relative Importance')
    plt.show()

    if debug:
        print(f'Top {n_importance} features: {sortedIdx}')
