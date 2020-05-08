import gzip
import wget
import glob
import os
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
        name +=  ".tsv"
        with open(name, 'wb') as out_file:
            out_file.write(s)
            inF.close()
        os.remove(file)
