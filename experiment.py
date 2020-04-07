import xenaPython as xena

GENES = ['FOXM1', 'TP53']

def get_codes(host, dataset, fields, data):
    "get codes for enumerations"
    codes = xena.field_codes(host, dataset, fields)
    codes_idx = dict([(x['name'], x['code'].split('\t')) for x in codes if x['code'] is not None])
    for i in range(len(fields)):
        if fields[i] in codes_idx:
            data[i] = [None if v == 'NaN' else codes_idx[fields[i]][int(v)] for v in data[i]]
    return data

def get_fields(host, dataset, samples, fields):
    "get field values"
    data = xena.dataset_fetch(host, dataset, samples, fields)
    return data

def get_fields_and_codes(host, dataset, samples, fields):
    "get fields and resolve codes"
    return get_codes( host, dataset, fields, get_fields( host, dataset, samples, fields))


hub = "https://pancanatlas.xenahubs.net"
cohort = 'TCGA PanCanAtlas'
host = xena.PUBLIC_HUBS['pancanAtlasHub']

# get samples in cohort
samples = xena.cohort_samples(host, cohort, None)
samples[0: 10]