import os
import itertools
from tqdm import tqdm

from april.fs import BPIC_DIR
from april.fs import EVENTLOG_DIR
from april.fs import EventLogFile
from april.fs import get_event_log_files
from april.generation import CategoricalAttributeGenerator
from april.generation.anomaly import *
from april.processmining.log import EventLog

xes_files = [
    'large_log.xes.gz',
    'small_log.xes.gz'
]

json_files = [
    'largelog-0.0-0.json.gz',
    'smalllog-0.0-0.json.gz'
]

for xes_file, json_file in tqdm(list(zip(xes_files, json_files))):
    event_log = EventLog.from_xes(os.path.join(BPIC_DIR, xes_file))
    event_log.save_json(os.path.join(EVENTLOG_DIR, json_file))
    

#Add anomalies

for k in range(0,10):
    np.random.seed(k) # This will ensure reproducibility
    ps = [0.3]
    event_log_paths = [e.path for e in get_event_log_files(EVENTLOG_DIR) if 'log' in e.name and e.p == 0.0]

    combinations = list(itertools.product(event_log_paths, ps))
    for event_log_path, p in tqdm(combinations, desc='Add anomalies'):
        event_log_file = EventLogFile(event_log_path)
        event_log = EventLog.from_json(event_log_path)

        anomalies = [
            ReplaceAnomaly(max_replacements=1)
        ]
        
        for anomaly in anomalies:
            # This is necessary to initialize the likelihood graph correctly
            anomaly.activities = event_log.unique_activities
            #anomaly.attributes = [CategoricalAttributeGenerator(name=name, values=values) for name, values in
           #                       event_log.unique_attribute_values.items() if name != 'name']        


        for case in tqdm(event_log):
            if np.random.uniform(0, 1) <= p:
                anomaly = np.random.choice(anomalies)
                anomaly.apply_to_case(case)
            else:
                NoneAnomaly().apply_to_case(case)

        event_log.save_json(str(EVENTLOG_DIR / f'{event_log_file.model}-{p}-{k}.json.gz'))
        
#Transform json to csv 

import gzip
import json
import pandas as pd
import os
from april.fs import EVENTLOG_DIR

json_files = [
    'smalllog-0.3-0.json.gz',
    'smalllog-0.3-1.json.gz',
    'smalllog-0.3-2.json.gz',
    'smalllog-0.3-3.json.gz',
    'smalllog-0.3-4.json.gz',
    'smalllog-0.3-5.json.gz',
    'smalllog-0.3-6.json.gz',
    'smalllog-0.3-7.json.gz',
    'smalllog-0.3-8.json.gz',
    'smalllog-0.3-9.json.gz',
    'largelog-0.3-0.json.gz',
    'largelog-0.3-1.json.gz',
    'largelog-0.3-2.json.gz',
    'largelog-0.3-3.json.gz',
    'largelog-0.3-4.json.gz',
    'largelog-0.3-5.json.gz',
    'largelog-0.3-6.json.gz',
    'largelog-0.3-7.json.gz',
    'largelog-0.3-8.json.gz',
    'largelog-0.3-9.json.gz'
]
csv_files = [
    'smalllog-0.3-0.csv',
    'smalllog-0.3-1.csv',
    'smalllog-0.3-2.csv',
    'smalllog-0.3-3.csv',
    'smalllog-0.3-4.csv',
    'smalllog-0.3-5.csv',
    'smalllog-0.3-6.csv',
    'smalllog-0.3-7.csv',
    'smalllog-0.3-8.csv',
    'smalllog-0.3-9.csv',
    'largelog-0.3-0.csv',
    'largelog-0.3-1.csv',
    'largelog-0.3-2.csv',
    'largelog-0.3-3.csv',
    'largelog-0.3-4.csv',
    'largelog-0.3-5.csv',
    'largelog-0.3-6.csv',
    'largelog-0.3-7.csv',
    'largelog-0.3-8.csv',
    'largelog-0.3-9.csv'
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


    
