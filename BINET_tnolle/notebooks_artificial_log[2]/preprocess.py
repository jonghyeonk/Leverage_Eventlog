from tqdm import tqdm

from april.fs import get_process_model_files
from april.generation.anomaly import *
from april.generation.utils import generate_for_process_model


#Define anomalies

anomalies = [
    SkipSequenceAnomaly(max_sequence_size=2),
    ReworkAnomaly(max_distance=5, max_sequence_size=3),
    EarlyAnomaly(max_distance=5, max_sequence_size=2),
    LateAnomaly(max_distance=5, max_sequence_size=2),
    InsertAnomaly(max_inserts=2)
]

#Generate Datasets

process_models = [m for m in get_process_model_files() if 'testing' not in m and 'paper' not in m]
for process_model in tqdm(process_models, desc='Generate'):
    generate_for_process_model(process_model, size=5000, anomalies=anomalies, num_attr=[0, 0,0,0,0, 0,0,0,0,0], seed=1337)

    
#Transform json to csv format

import gzip
import json
import pandas as pd
import os
from april.fs import EVENTLOG_DIR

json_files = [
    'huge-0.3-1.json.gz',
    'large-0.3-1.json.gz',
    'medium-0.3-1.json.gz',
    'small-0.3-1.json.gz',
    'wide-0.3-1.json.gz',
    'huge-0.3-2.json.gz',
    'large-0.3-2.json.gz',
    'medium-0.3-2.json.gz',
    'small-0.3-2.json.gz',
    'wide-0.3-2.json.gz',
    'huge-0.3-3.json.gz',
    'large-0.3-3.json.gz',
    'medium-0.3-3.json.gz',
    'small-0.3-3.json.gz',
    'wide-0.3-3.json.gz',
    'huge-0.3-4.json.gz',
    'large-0.3-4.json.gz',
    'medium-0.3-4.json.gz',
    'small-0.3-4.json.gz',
    'wide-0.3-4.json.gz',
    'huge-0.3-5.json.gz',
    'large-0.3-5.json.gz',
    'medium-0.3-5.json.gz',
    'small-0.3-5.json.gz',
    'wide-0.3-5.json.gz',
    'huge-0.3-6.json.gz',
    'large-0.3-6.json.gz',
    'medium-0.3-6.json.gz',
    'small-0.3-6.json.gz',
    'wide-0.3-6.json.gz',
    'huge-0.3-7.json.gz',
    'large-0.3-7.json.gz',
    'medium-0.3-7.json.gz',
    'small-0.3-7.json.gz',
    'wide-0.3-7.json.gz',
    'huge-0.3-8.json.gz',
    'large-0.3-8.json.gz',
    'medium-0.3-8.json.gz',
    'small-0.3-8.json.gz',
    'wide-0.3-8.json.gz',
    'huge-0.3-9.json.gz',
    'large-0.3-9.json.gz',
    'medium-0.3-9.json.gz',
    'small-0.3-9.json.gz',
    'wide-0.3-9.json.gz',
    'huge-0.3-10.json.gz',
    'large-0.3-10.json.gz',
    'medium-0.3-10.json.gz',
    'small-0.3-10.json.gz',
    'wide-0.3-10.json.gz'
]
csv_files = [
    'huge-0.3-1.csv',
    'large-0.3-1.csv',
    'medium-0.3-1.csv',
    'small-0.3-1.csv',
    'wide-0.3-1.csv',
    'huge-0.3-2.csv',
    'large-0.3-2.csv',
    'medium-0.3-2.csv',
    'small-0.3-2.csv',
    'wide-0.3-2.csv',
    'huge-0.3-3.csv',
    'large-0.3-3.csv',
    'medium-0.3-3.csv',
    'small-0.3-3.csv',
    'wide-0.3-3.csv',
    'huge-0.3-4.csv',
    'large-0.3-4.csv',
    'medium-0.3-4.csv',
    'small-0.3-4.csv',
    'wide-0.3-4.csv',
    'huge-0.3-5.csv',
    'large-0.3-5.csv',
    'medium-0.3-5.csv',
    'small-0.3-5.csv',
    'wide-0.3-5.csv',
    'huge-0.3-6.csv',
    'large-0.3-6.csv',
    'medium-0.3-6.csv',
    'small-0.3-6.csv',
    'wide-0.3-6.csv',
    'huge-0.3-7.csv',
    'large-0.3-7.csv',
    'medium-0.3-7.csv',
    'small-0.3-7.csv',
    'wide-0.3-7.csv',
    'huge-0.3-8.csv',
    'large-0.3-8.csv',
    'medium-0.3-8.csv',
    'small-0.3-8.csv',
    'wide-0.3-8.csv',
    'huge-0.3-9.csv',
    'large-0.3-9.csv',
    'medium-0.3-9.csv',
    'small-0.3-9.csv',
    'wide-0.3-9.csv',
    'huge-0.3-10.csv',
    'large-0.3-10.csv',
    'medium-0.3-10.csv',
    'small-0.3-10.csv',
    'wide-0.3-10.csv'
]
for json_file, csv_file in tqdm(list(zip(json_files, csv_files))):
    with gzip.GzipFile(os.path.join(EVENTLOG_DIR, json_file), 'r') as fin:    # 4. gzip
        json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)
    json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
    data = json.loads(json_str)                      # 1. data
    df= pd.DataFrame()
    for case in data['cases']:
        pos=0
        for e in case['events']:
            pos+=1
            event = pd.DataFrame([{'caseid':case['id'], 'order':pos,**case['attributes'],'name':e['name'], 'timestamp':e['timestamp']}])
            event = pd.DataFrame([{'order':pos,**case['attributes'],'caseid':case['id'], 'name':e['name'], 'timestamp':e['timestamp']}])
            df=df.append(event, ignore_index=True)
    df.to_csv(os.path.join(EVENTLOG_DIR, csv_file) ,index=False)
    
