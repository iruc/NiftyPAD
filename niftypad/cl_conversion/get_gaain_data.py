
import requests
import urllib.request
import zipfile
import io
import numpy as np
import pandas as pd
import glob
import os
import nibabel as nib



def get_supplementary_tables():
    urllib.request.urlretrieve("https://www.gaaindata.org/data/centiloid/SupplementaryTable1.xlsx")
    file = urllib.request.urlopen("https://www.gaaindata.org/data/centiloid/SupplementaryTable1.xlsx")
    #table1_supplement = pd.read_excel()
    tables_path = "https://www.gaaindata.org/data/centiloid/SupplementaryTable1.xlsx"
    supplementary_table1 = pd.read_excel(tables_path)
    ad_rows = supplementary_table1['Supplementary Table 1.'].str.contains('AD')
    yc_rows = supplementary_table1['Supplementary Table 1.'].str.contains('YC')
    suvr_df = pd.DataFrame()
    reformed_table = supplementary_table1.drop([0,1,2,3,49], axis=0).reset_index(drop=True)
    table_subjects = [value.strip() for value in reformed_table['Supplementary Table 1.'].values]
    reformed_table['Supplementary Table 1.'] = table_subjects
    return reformed_table


def get_list_subjects(path):
    subjects = sorted(glob.glob(os.path.join(path, '*.nii')))
    list_id = []
    for subject in subjects:
        base_id = os.path.basename(subject)
        id = base_id.split('_')
        list_id.append(id[0])

    return list_id, subjects


def get_ad_yc_subjects(list_path=None, subjects_path=[]):
    if list_path is None:
        yc_path = subjects_path[0]
        list_yc, _ = get_list_subjects(yc_path)
        ad_path = subjects_path[1]
        list_ad, _ = get_list_subjects(ad_path)
        classif_ad = ['AD']*len(list_ad)
        classif_yc = ['YC']*len(list_ad)
        df_subjects = pd.DataFrame(data={'subjects':np.concatenate([list_ad, list_yc]),
                                         'group': np.concatenate([classif_ad, classif_yc])})
    else:
        df_subjects = pd.read_excel(list_path)

    return df_subjects


def get_mni_pet(path):
    path = 'YC-0_MR/nifti'
    list_id_ad, subjects = get_list_subjects(path)
    subjects_data = []
    for subject in subjects:
        nifti_subject = nib.load(subject)
        subjects_data.append(np.array(nifti_subject.dataobj))

    list_id_yc, subjects = get_list_subjects(path)
    for subject in subjects:
        nifti_subject = nib.load(subject)
        subjects_data.append(np.array(nifti_subject.dataobj))
    full_list = np.concatenate([list_id_ad, list_id_yc])
    return subjects_data, full_list


def download_subjects_gaain():
    zip_file_url = "https://www.gaaindata.org/data/centiloid/Centiloid_Std_VOI.zip"
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    centiloid_voi = zipfile.ZipFile.extract(z, member='Centiloid_Std_VOI')


def download_subjects_gaain():
    zip_file_url = "https://www.gaaindata.org/data/centiloid/YC-0_MR.zip"
    rr = requests.get(zip_file_url)
    zz = zipfile.ZipFile(io.BytesIO(rr.content))
    zz.extractall()


