import gzip
import wget
import glob
import os
from pathlib import Path

Path("./data/").mkdir(parents=True, exist_ok=True)

if not os.path.exists('./data/TCGA-LUAD.tsv') and not os.path.exists('./data/TCGA-LUSC.tsv'):
    luad = "https://gdc.xenahubs.net/download/TCGA-LUAD.mutect2_snv.tsv.gz"
    lusc = "https://gdc.xenahubs.net/download/TCGA-LUSC.mutect2_snv.tsv.gz"

    print("\nDownloading luad")
    wget.download(luad, "./data/")
    print("\nDownloading lusc")
    wget.download(lusc, "./data/")
    print("\nDone downloading...")
    print(".\n.\n.\n.")
    print("Extracting")
    for file in glob.glob(os.path.join("./data/", '*.gz')):
        inF = gzip.open(file, 'rb')
        s = inF.read()
        name = "./data/"+ (file.split("/")[2].split(".")[0]) + ".tsv"
        with open(name, 'wb') as out_file:
            out_file.write(s)
            inF.close()
        os.remove(file)
