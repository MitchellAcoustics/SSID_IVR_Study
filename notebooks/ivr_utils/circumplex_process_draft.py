#%%

import csv
from pathlib import Path
import pandas as pd
from datetime import datetime
import soundscapy as sspy
from loguru import logger

#%%

def proc_single_csv(csv_filepath):
    # Read in the data
    with open(csv_filepath, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        dictobj = next(csv_reader)
    logger.info(f"Processing {csv_filepath}")
    dictobj['scene_id'] = list(dictobj.keys())[-1]
    dictobj = {k: v for k, v in dictobj.items() if v != 'NULL'}
    return dictobj

def proc_exp_dir(exp_dir):
    res = [proc_single_csv(csv) for csv in exp_dir.glob("*.csv")]
    res_df = pd.DataFrame(res)
    return res_df

def proc_part_dir(part_dir):
    part_res = {folder.name: proc_exp_dir(folder) for folder in part_dir.glob('*') if folder.name != '.DS_Store'}
    return part_res


def proc_circ_dir(circ_dir, index_file='SSID IVR STUDY 1 EXPERIMENT INDEX.xlsx'):

    index_df = pd.read_excel(index_file, sheet_name='Participants')
    valid_participants = index_df[index_df['CIRCUMPLEX DATA CHECKED'] == 'Yes']['PARTICIPANT\'S CODE'].to_list()

    a_dict = {}
    av_dict = {}
    v_dict = {}
    errors = []
    for folder in circ_dir.glob('*'):
        if folder.name != '.DS_Store':
            if folder.name in valid_participants:
                try:
                    part_res = proc_part_dir(folder)
                    a_dict[folder.name] = part_res[f'{folder.name} A']
                    av_dict[folder.name] = part_res[f'{folder.name} AV'] if f'{folder.name} AV' in part_res.keys() else part_res[f'{folder.name} AV_corrected']
                    v_dict[folder.name] = part_res[f'{folder.name} V'] if f'{folder.name} V' in part_res.keys() else part_res[f'{folder.name} V_corrected']
                except:
                    print(f"Error in {folder.name}")
                    errors.append(folder.name)

    def add_session_id(df, index_file):
        sessionid_df = pd.read_excel(index_file, sheet_name='SessionIDs', index_col='UNITY SCENE')
        df['SessionID'] = df['scene_id'].map(sessionid_df['SessionID'])
        return df
    
    def sspy_cal_iso(df):
        df_copy = df.copy()
        paq_aliases = {
            'Pleasant': 'PAQ1',
            'Vibrant': 'PAQ2',
            'Eventful': 'PAQ3',
            'Chaotic': 'PAQ4',
            'Annoying': 'PAQ5',
            'Monotonous': 'PAQ6',
            'Uneventful': 'PAQ7',
            'Calm': 'PAQ8',
        }
        df_copy.rename(paq_aliases, inplace=True, axis=1)
        df_copy = sspy.surveys.return_paqs(df_copy, incl_ids = False)
        df_copy = df_copy.astype(float)
        ISOPleasant, ISOEventful = sspy.surveys.processing.calculate_iso_coords(df_copy, val_range=(-50, 50))
        return ISOPleasant, ISOEventful
    
    def create_df(exp_dict, index_file):
        exp_df = pd.concat(exp_dict)
        exp_df.index = exp_df.index.set_names(['Participant', 'trial'])
        exp_df.reset_index(inplace=True)
        exp_df = add_session_id(exp_df, index_file)
        exp_pl, exp_ev = sspy_cal_iso(exp_df)
        exp_df['ISOPleasant'] = exp_pl
        exp_df['ISOEventful'] = exp_ev
        return exp_df

    a_df = create_df(a_dict, index_file)
    av_df = create_df(av_dict, index_file)
    v_df = create_df(v_dict, index_file)
    errors_df = pd.DataFrame(errors, columns=['participant'])

    return a_df, av_df, v_df, errors_df

def save_circ_data(circ_dir, out_file=None, index_file='SSID IVR STUDY 1 EXPERIMENT INDEX.xlsx'):
    if out_file is None:
        out_file = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_circumplex_data.xlsx"

    a_res, av_res, v_res, errors = proc_circ_dir(circ_dir, index_file=index_file)
    writer = pd.ExcelWriter(out_file, engine='xlsxwriter')
    logger.info(f"Saving circumplex data to {out_file}")
    a_res.to_excel(writer, sheet_name='A', index=False)
    av_res.to_excel(writer, sheet_name='AV', index=False)
    v_res.to_excel(writer, sheet_name='V', index=False)
    errors.to_excel(writer, sheet_name='Processing Errors', index=False)
    writer.close()

    return a_res, av_res, v_res, errors



# %%
# single_csv_file = 'Circumplex Data/P1/P1 A/MyQuestionnaire_30.csv'
# p1a = proc_single_csv(single_csv_file)

if __name__ == '__main__':
    from loguru import logger
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    root_path = Path("/Users/mitch/Library/CloudStorage/OneDrive-UniversityCollegeLondon/_Fellowship/Projects/SSID Segmentation/Circumplex Data")
    circ_dir = root_path.joinpath('Circumplex Data')

    a_res, av_res, v_res, errors = save_circ_data(circ_dir, index_file=root_path.joinpath('SSID IVR STUDY 1 EXPERIMENT INDEX.xlsx'))



#%%

