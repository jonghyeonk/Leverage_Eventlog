
#######################################################################################################





import itertools
from multiprocessing.pool import Pool

import pandas as pd
from sklearn import metrics
from sqlalchemy.orm import Session
from tqdm import tqdm

from april.anomalydetection import BINet
from april.anomalydetection.utils import label_collapse
from april.database import Evaluation
from april.database import Model
from april.database import get_engine
from april.enums import Base
from april.enums import Heuristic
from april.enums import Strategy
from april.evaluator import Evaluator
from april.fs import get_model_files
from april.fs import PLOT_DIR


heuristics = [h for h in Heuristic.keys() if h not in [Heuristic.DEFAULT, Heuristic.MANUAL, Heuristic.RATIO,
                                                       Heuristic.MEDIAN, Heuristic.MEAN]]
params = [(Base.SCORES, Heuristic.DEFAULT, Strategy.SINGLE), *itertools.product([Base.SCORES], heuristics, Strategy.keys())]


def _evaluate(params):
    e, base, heuristic, strategy = params

    session = Session(get_engine())
    model = session.query(Model).filter_by(file_name=e.model_file.name).first()
    session.close()

    # Generate evaluation frames
    y_pred = e.binarizer.binarize(base=base, heuristic=heuristic, strategy=strategy, go_backwards=False)
    y_true = e.binarizer.get_targets()

    evaluations = []
    for axis in [0, 1, 2]:
        for i, attribute_name in enumerate(e.dataset.attribute_keys):
            def get_evaluation(label, precision, recall, f1):
                return Evaluation(model_id=model.id, file_name=model.file_name,
                                  label=label, perspective=perspective, attribute_name=attribute_name,
                                  axis=axis, base=base, heuristic=heuristic, strategy=strategy,
                                  precision=precision, recall=recall, f1=f1)

            perspective = 'Control Flow' if i == 0 else 'Data'
            if i > 0  and not e.ad_.supports_attributes:
                evaluations.append(get_evaluation('Normal', 0.0, 0.0, 0.0))
                evaluations.append(get_evaluation('Anomaly', 0.0, 0.0, 0.0))
            else:
                yp = label_collapse(y_pred[:, :, i:i + 1], axis=axis).compressed()
                yt = label_collapse(y_true[:, :, i:i + 1], axis=axis).compressed()
                p, r, f, _ = metrics.precision_recall_fscore_support(yt, yp, labels=[0, 1])
                evaluations.append(get_evaluation('Normal', p[0], r[0], f[0]))
                evaluations.append(get_evaluation('Anomaly', p[1], r[1], f[1]))

    return evaluations

def evaluate(model_name):
    e = Evaluator(model_name)

    _params = []
    for base, heuristic, strategy in params:
        if e.dataset.num_attributes == 1 and strategy in [Strategy.ATTRIBUTE, Strategy.POSITION_ATTRIBUTE]:
            continue
        if isinstance(e.ad_, BINet) and e.ad_.version == 0:
            continue
        if heuristic is not None and heuristic not in e.ad_.supported_heuristics:
            continue
        if strategy is not None and strategy not in e.ad_.supported_strategies:
            continue
        if base is not None and base not in e.ad_.supported_bases:
            continue
        _params.append([e, base, heuristic, strategy])

    return [_e for p in _params for _e in _evaluate(p)]


models = sorted([m.name for m in get_model_files() ])

evaluations = []
for i in range(len(models)):
    e= evaluate(models[i])
    evaluations.append(e)

# Write to database
session = Session(get_engine())
for e in evaluations:
    session.bulk_save_objects(e)
    session.commit()
session.close()


out_dir = PLOT_DIR / 'isj-2019'
eval_file = out_dir / 'eval.pkl'

session = Session(get_engine())
evaluations = session.query(Evaluation).all()
rows = []
for ev in tqdm(evaluations):
    m = ev.model
    el = ev.model.training_event_log
    rows.append([m.file_name, m.creation_date, m.hyperparameters, m.training_duration, m.training_host, m.algorithm,
                 m.file_name.split('-')[0], m.file_name, m.creation_date, m.hyperparameters,
                 ev.axis, ev.base, ev.heuristic, ev.strategy, ev.label, ev.attribute_name, ev.perspective, ev.precision, ev.recall, ev.f1])
session.close()
columns = ['file_name', 'date', 'hyperparameters', 'training_duration', 'training_host', 'ad',
           'dataset_name', 'process_model', 'noise', 'dataset_id',
           'axis', 'base', 'heuristic', 'strategy', 'label', 'attribute_name', 'perspective', 'precision', 'recall', 'f1']

evaluation = pd.DataFrame(rows, columns=columns)
evaluation.to_pickle(eval_file)



############################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy.orm import Session
import scikit_posthocs as sp

from april.database import get_engine
from april.fs import PLOT_DIR
from april.utils import microsoft_colors, prettify_dataframe, cd_plot, get_cd
from april.enums import Base, Strategy, Heuristic

sns.set_style('white')
pd.set_option('display.max_rows', 50)

out_dir = PLOT_DIR / 'isj-2019'
eval_file = out_dir / 'eval.pkl'


synth_datasets = ['small', 'medium', 'large', 'huge', 'wide']
#bpic_datasets = ['bpic12', 'bpic13', 'bpic15', 'bpic17', 'largelog','smalllog']
bpic_datasets = ['bpic12', 'bpic13', 'bpic15', 'bpic17']

datasets = synth_datasets + bpic_datasets
dataset_types = ['Synthetic', 'Real-life']

binet_ads = ["BINetv1"]
nn_ads = ["DAE"] + binet_ads
d_ads = ["Naive", "Sampling","DAE","BINetv1", "OC-SVM"]
ads = nn_ads + d_ads

heuristics = [r'$best$', r'$default$', r'$elbow_\downarrow$', r'$elbow_\uparrow$',
              r'$lp_\leftarrow$', r'$lp_\leftrightarrow$', r'$lp_\rightarrow$']

evaluation = pd.read_pickle(eval_file)

evaluation = evaluation.query(f'ad in {ads} and label == "Anomaly"')

evaluation['perspective-label'] = evaluation['perspective'] + '-' + evaluation['label']
evaluation['attribute_name-label'] = evaluation['attribute_name'] + '-' + evaluation['label']
evaluation['dataset_type'] = 'Synthetic'
evaluation.loc[evaluation['process_model'].str.contains('bpic'), 'dataset_type'] = 'Real-life'
evaluation.loc[evaluation['process_model'].str.contains('real'), 'dataset_type'] = 'Real-life'

_filtered_evaluation = evaluation.query(f'ad in {nn_ads} and (strategy == "{Strategy.ATTRIBUTE}"'
                                       f' or (strategy == "{Strategy.SINGLE}" and process_model == "bpic12")'
                                       f' or (strategy == "{Strategy.SINGLE}" and ad == "Naive+")) or ad in {d_ads}')

filtered_evaluation = _filtered_evaluation.query(f'heuristic == "{Heuristic.DEFAULT}"'
                                                 f' or (heuristic == "{Heuristic.LP_MEAN}" and ad != "DAE")'
                                                 f' or (heuristic == "{Heuristic.ELBOW_UP}" and ad == "DAE")')

df = filtered_evaluation.query('axis in [0, 2]')
df = prettify_dataframe(df)
df = df.groupby(['axis', 'process_model', 'dataset_name', 'ad', 'file_name', 'perspective'])['precision', 'recall', 'f1'].mean().reset_index()
df = df.groupby(['axis', 'process_model', 'dataset_name', 'ad', 'file_name'])['precision', 'recall', 'f1'].mean().reset_index()
df['f1'] = 2 * df['recall'] * df['precision'] / (df['recall'] + df['precision'])

df = pd.pivot_table(df, index=['axis', 'ad'], columns=['process_model', 'dataset_name'], values=['precision', 'recall', 'f1'])
df = df.fillna(0)
df = df.stack(1).stack(1).reset_index()
df.to_excel(str(out_dir / 'table.xlsx'), index=False)

df = pd.pivot_table(df, index=['axis', 'ad'], columns=['process_model'], values=['f1'], aggfunc=np.mean)
df.round(2)


