import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from sklearn.metrics import r2_score
from lib.inference import remove_nan

GROUND_TRUTH_CSV = r"D:\Local\APAYN\dicoms\sz.csv"
PHILIPS_CSV = "./data/predictions_Philips Medical Systems.csv"
SIEMENS_CSV = "./data/predictions_SIEMENS.csv"

COLUMN_MAPS = {
    "LVEDV": "lv_cav",
    "LVm": "lv_wall",
    "RVEDV": "rv_cav"
}

df_ground_truth = pd.read_csv(GROUND_TRUTH_CSV)
df_philips = pd.read_csv(PHILIPS_CSV)
df_siemens = pd.read_csv(SIEMENS_CSV)

true_pred_by_measure_by_manu = defaultdict(lambda: defaultdict(list))

for manu, df_manu in zip(('philips',), (df_philips,)):
#for manu, df_manu in zip(('philips', 'siemens'), (df_philips, df_siemens)):
    for i_row, row_pred in df_manu.iterrows():
        study_id = row_pred['study_id']

        if study_id not in list(df_ground_truth['accession_number']):
            continue

        row_true = df_ground_truth.loc[df_ground_truth['accession_number'] == study_id]
        for truth_colname, pred_colname in COLUMN_MAPS.items():
            try:
                pred_val = float(row_pred[pred_colname])
                true_val = float(row_true[truth_colname])

                true_pred_by_measure_by_manu[truth_colname][manu].append((true_val, pred_val))
            except ValueError:
                continue

for measure, measure_dict in true_pred_by_measure_by_manu.items():
    y_true, y_pred, manu = [], [], []
    for manufacturer, list_of_measures in measure_dict.items():
        y_t, y_p = zip(*list_of_measures)
        y_t, y_p = remove_nan(y_t, y_p)

        y_true.extend(y_t)
        y_pred.extend(y_p)
        manu.extend([manufacturer] * len(y_t))

    r2 = r2_score(y_true, y_pred)
    max_ = max(max(y_true), max(y_pred))
    plt.scatter(y_true, y_pred, alpha=0.3, c=['red' if m == 'siemens' else 'blue' for m in manu])
    plt.title(f"{measure}")
    siemens_patch = mpatches.Patch(color='red', label='Siemens')
    philips_patch = mpatches.Patch(color='blue', label='Philips')
    plt.legend(handles=[siemens_patch, philips_patch])
    plt.plot([0, max_], [0, max_], 'k--')
    plt.show()
