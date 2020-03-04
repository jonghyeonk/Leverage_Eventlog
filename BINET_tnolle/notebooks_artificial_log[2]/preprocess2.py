
import gzip
import json
import pandas as pd
import os
from april.fs import EVENTLOG_DIR
from tqdm import tqdm


json_files = [
    'bpic13-0.3-1.json.gz',
    'bpic13-0.3-2.json.gz',
    'bpic13-0.3-3.json.gz',
    'bpic15-0.3-1.json.gz',
    'bpic15-0.3-2.json.gz',
    'bpic15-0.3-3.json.gz',
    'bpic15-0.3-4.json.gz',
    'bpic15-0.3-5.json.gz',
    'bpic12-0.3-0.json.gz',
    'bpic17-0.3-1.json.gz',
    'bpic17-0.3-2.json.gz'
]

csv_files = [
    'bpic13-0.3-1.csv',
    'bpic13-0.3-2.csv',
    'bpic13-0.3-3.csv',
    'bpic15-0.3-1.csv',
    'bpic15-0.3-2.csv',
    'bpic15-0.3-3.csv',
    'bpic15-0.3-4.csv',
    'bpic15-0.3-5.csv',
    'bpic12-0.3-0.csv',
    'bpic17-0.3-1.csv',
    'bpic17-0.3-2.csv'
]
for json_file, csv_file in tqdm(list(zip(json_files, csv_files))):
    print(json_file)
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

