import gzip
import wget
import glob
import os
from pathlib import Path

Path("./data/").mkdir(parents=True, exist_ok=True)

luad = "https://gdc.xenahubs.net/download/TCGA-LUAD.htseq_counts.tsv.gz"
lusc = "https://gdc.xenahubs.net/download/TCGA-LUSC.htseq_counts.tsv.gz"

print("Downloading luad")
wget.download(luad, "./data/")
print("Downloading lusc")
wget.download(lusc, "./data/")
print("Done downloading...")
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