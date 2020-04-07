import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xenaPython as xena

# cohort = 'GDC TCGA Lung Adenocarcinoma'
# host = xena.PUBLIC_HUBS['gdcHub']
# samples = xena.cohort_samples(host, cohort, None)
# print(samples)

hub = "https://gdc.xenahubs.net"
dataset = "TCGA-LUAD.htseq_counts.tsv"
df = pd.read_csv(dataset, sep='\t', header=0)
# Remove columns with missing NaN values
# df.dropna(inplace=True)
# print(df.shape)
# # colnames = df.columns
# # print(colnames)
print(df.shape)

# LUAD

# identifiers = xena.dataset_field(hub, dataset)
# print(identifiers)
# num_identifiers = xena.dataset_field_n (hub, dataset)
# print(num_identifiers)
# last term specifies the limit - optional
# samples = xena.dataset_samples (hub, dataset, 10)


# # colnames = df.columns
# # print(colnames)
# print(df.head())

# samples = xena.dataset_samples (hub, dataset, 10)


# hub = "https://gdc.xenahubs.net"
# dataset = "TCGA-LUAD.htseq_counts.tsv"
# probes = 
