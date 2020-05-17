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
import pickle
import glob, os
from os.path import exists
import gzip
import wget
from sys import argv
from pathlib import Path

def download_tsv(link, name):
    """
    Download files from xena for exploration, extract and save as name.tsv

    Arguments:
        link {str} -- [Link to download the file from]
        name {str} -- [save data as name.tsv]
    """

    Path("./data/").mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {link}")
    wget.download(link, "./data/")
    print("\nDone downloading...")
    print(".\n.\n.\n.")
    print("Extracting")
    for file in glob.glob(os.path.join("./data/", '*.gz')):
        inF = gzip.open(file, 'rb')
        s = inF.read()
        name = "./data/"+name
        with open(name, 'wb') as out_file:
            out_file.write(s)
            inF.close()
        os.remove(file)
    print("Extraction done")

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


def save_to_pickle(df, filename, standardise=False):
    """
    Given a dataframe, save it to pickle for faster fetch

    Arguments:
        df {pd.DataFrame} -- Already loaded dataframe
        filename {string} -- Filename to save it as

    Keyword Arguments:
        standardise {bool} -- If you need to standardise the data or not (default: {False})
    """

    if standardise:
        df = pd.DataFrame(StandardScaler().fit_transform(df), columns = df.columns)

    filename = filename if "." in filename else filename+".pkl"
    pickle.dump(df, open(f"data/{filename}", "wb"))

def cleanup_tsv():
    path = "data/"
    if not exists(path):
        print("Nothing to clean up")
        return

    for file in glob.glob(os.path.join("./data/", '*.tsv')):
        df = load_and_fillna(file)
        filename = file.replace("tsv", "pkl")
        filename = filename.split("/")[-1].split("-")[-1]
        save_to_pickle(df, filename)
        save_to_pickle(df, "std_"+filename ,standardise=True)
        os.remove(file)



def main():
    files = {
        "LUSC" : "https://gdc.xenahubs.net/download/TCGA-LUSC.htseq_counts.tsv.gz",
        "LUAD" : "https://gdc.xenahubs.net/download/TCGA-LUAD.htseq_counts.tsv.gz",
        "KIRP" : "https://gdc.xenahubs.net/download/TCGA-KIRP.htseq_counts.tsv.gz",
        "KIRC" : "https://gdc.xenahubs.net/download/TCGA-KIRC.htseq_counts.tsv.gz"
    }

    for file,link in files.items():
        path = f"data/{file}.pkl"
        if not exists(path):
            download_tsv(link, file+".tsv")
            print("Converting to pickle")
            df = load_and_fillna(path.replace("pkl", "tsv"))
            save_to_pickle(df, file+".pkl")
            save_to_pickle(df, "std_"+file+".pkl", standardise=True)
            cleanup_tsv()

    print("Done")
if __name__ == "__main__":
    main()
