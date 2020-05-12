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
        name = "./data/"+name
        with open(name, 'wb') as out_file:
            out_file.write(s)
            inF.close()
        os.remove(file)
    print("Extraction done")
    print(os.listdir("./data/"))

if __name__ == "__main__":
    link = str(input("Link to the zipped file: "))
    name = str(input("Filename to save as, include extensions such as .tsv: "))
    download_tsv(link=link, name=name)
