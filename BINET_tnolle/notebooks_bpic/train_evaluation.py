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
    'BPIC15_4.xes.gz',
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



np.random.seed(0)  # This will ensure reproducibility
ps = [0.3]
event_log_paths = [e.path for e in get_event_log_files(EVENTLOG_DIR) if 'bpic' in e.name and e.p == 0.0]

combinations = list(itertools.product(event_log_paths, ps))
for event_log_path, p in tqdm(combinations, desc='Add anomalies'):
    event_log_file = EventLogFile(event_log_path)
    event_log = EventLog.from_json(event_log_path)

    anomalies = [
        ReplaceAnomaly(max_replacements=1),
        SkipSequenceAnomaly(max_sequence_size=2),
        ReworkAnomaly(max_distance=5, max_sequence_size=3),
        EarlyAnomaly(max_distance=5, max_sequence_size=2),
        LateAnomaly(max_distance=5, max_sequence_size=2),
        InsertAnomaly(max_inserts=2)
    ]

    for anomaly in anomalies:
        anomaly.activities = event_log.unique_activities

    for case in tqdm(event_log):
        if np.random.uniform(0, 1) <= p:
            anomaly = np.random.choice(anomalies)
            anomaly.apply_to_case(case)
        else:
            NoneAnomaly().apply_to_case(case)

    event_log.save_json(str(EVENTLOG_DIR / f'{event_log_file.model}-{p}-{event_log_file.id}.json.gz'))