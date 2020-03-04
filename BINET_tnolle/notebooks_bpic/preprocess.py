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
    'BPIC19.xes.gz',
    'BPIC13_closed_problems.xes.gz',
    'BPIC13_open_problems.xes.gz',
    'BPIC13_incidents.xes.gz',
    'BPIC15_1.xes.gz',
    'BPIC15_2.xes.gz',
    'BPIC15_3.xes.gz',
    'BPIC15_4.xes.gz'
    'BPIC15_5.xes.gz',
    'BPIC12.xes.gz',
    'BPIC17.xes.gz',
    'BPIC17_offer_log.xes.gz'
]

json_files = [
    'bpic19-0.0-0.json.gz',
    'bpic13-0.0-1.json.gz',
    'bpic13-0.0-2.json.gz',
    'bpic13-0.0-3.json.gz',
    'bpic15-0.0-1.json.gz',
    'bpic15-0.0-2.json.gz',
    'bpic15-0.0-3.json.gz',
    'bpic15-0.0-4.json.gz',
    'bpic15-0.0-5.json.gz',
    'bpic12-0.0-0.json.gz',
    'bpic17-0.0-1.json.gz',
    'bpic17-0.0-2.json.gz'
]

for xes_file, json_file in tqdm(list(zip(xes_files, json_files))):
    event_log = EventLog.from_xes(os.path.join(BPIC_DIR, xes_file))
    event_log.save_json(os.path.join(EVENTLOG_DIR, json_file))


#Add anomalies

np.random.seed(0)  # This will ensure reproducibility
ps = [0.3]
event_log_paths = [e.path for e in get_event_log_files(EVENTLOG_DIR) if 'bpic' in e.name and e.p == 0.0]

combinations = list(itertools.product(event_log_paths, ps))
for event_log_path, p in tqdm(combinations, desc='Add anomalies'):
    event_log_file = EventLogFile(event_log_path)
    event_log = EventLog.from_json(event_log_path)

    anomalies = [
        SkipSequenceAnomaly(max_sequence_size=2),
        ReworkAnomaly(max_distance=5, max_sequence_size=3),
        EarlyAnomaly(max_distance=5, max_sequence_size=2),
        LateAnomaly(max_distance=5, max_sequence_size=2),
        InsertAnomaly(max_inserts=2)
    ]

#     if event_log.num_event_attributes > 0:
#         anomalies.append(AttributeAnomaly(max_events=3, max_attributes=min(2, event_log.num_activities)))

    for anomaly in anomalies:
        # This is necessary to initialize the likelihood graph correctly
        anomaly.activities = event_log.unique_activities
#         anomaly.attributes = [CategoricalAttributeGenerator(name=name, values=values) for name, values in
                           #   event_log.unique_attribute_values.items() if name != 'name']

    for case in tqdm(event_log):
        if np.random.uniform(0, 1) <= p:
            anomaly = np.random.choice(anomalies)
            anomaly.apply_to_case(case)
           # print(dict(a.name, f'Random {a.name} {np.random.randint(1, len(a.values))}') for a in case.attributes)
        else:
            NoneAnomaly().apply_to_case(case)

    event_log.save_json(str(EVENTLOG_DIR / f'{event_log_file.model}-{p}-{event_log_file.id}.json.gz'))

    
    
#Transform json to csv format

import gzip
import json
import pandas as pd
import os
from april.fs import EVENTLOG_DIR

json_files = [
    'bpic19-0.0-0.json.gz',
    'bpic13-0.0-1.json.gz',
    'bpic13-0.0-2.json.gz',
    'bpic13-0.0-3.json.gz',
    'bpic15-0.0-1.json.gz',
    'bpic15-0.0-2.json.gz',
    'bpic15-0.0-3.json.gz',
    'bpic15-0.0-4.json.gz',
    'bpic15-0.0-5.json.gz',
    'bpic12-0.0-0.json.gz',
    'bpic17-0.0-1.json.gz',
    'bpic17-0.0-2.json.gz'
]

csv_files = [
    'bpic19-0.0-0.csv',
    'bpic13-0.0-1.csv',
    'bpic13-0.0-2.csv',
    'bpic13-0.0-3.csv',
    'bpic15-0.0-1.csv',
    'bpic15-0.0-2.csv',
    'bpic15-0.0-3.csv',
    'bpic15-0.0-4.csv',
    'bpic15-0.0-5.csv',
    'bpic12-0.0-0.csv',
    'bpic17-0.0-1.csv',
    'bpic17-0.0-2.csv'
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

