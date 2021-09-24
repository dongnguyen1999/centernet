import os
import shutil
from glob import glob
import zipfile
import pandas as pd

def auto_concat_rfds(path):
    names=['filename', 'x1', 'y1', 'x2', 'y2', 'label']
    output_path = os.path.join(path, 'output')
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for index, zip_file in enumerate(glob(os.path.join(path, '*.zip'))):
        print(f'Start unzip file {zip_file}')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        os.remove(os.path.join(output_path, 'README.roboflow.txt'))
        os.remove(os.path.join(output_path, 'README.dataset.txt'))
        os.rename(os.path.join(output_path, 'train', '_annotations.csv'), os.path.join(output_path, 'train', f'_annotations{index+1}.csv'))
    
    df = pd.DataFrame([])
    train_path = os.path.join(output_path, 'train')
    for csv_file in glob(os.path.join(train_path, '*.csv')):
        csv_df = pd.read_csv(csv_file, names=names)
        df = pd.concat([df, csv_df])
        print(f'Read {len(csv_df)} row from {csv_file}')
        os.remove(csv_file)

    valid_path = os.path.join(path, 'valid')
    shutil.copytree(valid_path, os.path.join(output_path, 'valid')) 

    print(f'Sum total row is {len(df)}')
    df.to_csv(os.path.join(train_path, '_annotations_custom_v2.txt'), index=False, header=False)
    

