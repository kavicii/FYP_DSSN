import os
import argparse
from tqdm import tqdm
from typing import Union
import numpy as np
import pandas as pd
import cv2
from preprocess import compute_features
from read_file import read_csv
import warnings

def save_CASME_preprocess_weight(data_info: pd.DataFrame, root: str):
    """Save the preprocess weight for the repeating usage

    Parameters
    ----------
    data_info : pd.DataFrame
        DataFrame for the data info
    root : str
        Root of the original image
    path : str
        Path to save the weight
    """
    print(data_info.memory_usage(deep=True))
    for idx in tqdm(range(len(data_info))):
        # Subject of the data folder
        subject = data_info.loc[idx, "Subject"]
        img_root = f"{root}\sub{subject}\{data_info.loc[idx, 'Filename']}\\"
        onset_frame_name = img_root + f"reg_img{data_info.loc[idx, 'Onset']}.jpg"
        apex_frame_name = img_root + f"reg_img{data_info.loc[idx, 'Apex']}.jpg"

        onset_frame = cv2.imread(onset_frame_name)
        onset_frame = cv2.cvtColor(onset_frame, cv2.COLOR_BGR2RGB)
        if onset_frame is None:
            print(onset_frame_name)

        apex_frame = cv2.imread(apex_frame_name)
        apex_frame = cv2.cvtColor(apex_frame, cv2.COLOR_BGR2RGB)
        if apex_frame is None:
            print(apex_frame_name)

        # Convert to wanted features
        flow, mag, strain, gray = compute_features(onset_frame, apex_frame)
        
        np.savez(file=f"{root}\sub{subject}\{data_info.loc[idx, 'Filename']}\{subject}_{data_info.loc[idx, 'Filename']}.npz",
                 flow=flow,
                 mag=mag,
                 gray=gray,
                 strain=strain)

if __name__ == "__main__":
    data, _ = read_csv('CASME2CSV.csv')
    root = "Dataset\Cropped"
    save_CASME_preprocess_weight(data_info=data,
                                     root=root)
    # Argument parse
#    parser = argparse.ArgumentParser()
#   parser.add_argument("--csv_path",
#                        type=str,
#                        required=True)
#    parser.add_argument("--root",
#                        type=str,
#                        required=True)
#    parser.add_argument("--catego",
#                        type=str,
#                        required=True)
#    args = parser.parse_args()

    # Save the weight
#    data, _ = read_csv(args.csv_path)
    
#    if args.catego == "CASME":
#        save_CASME_preprocess_weight(data_info=data,
#                                     root=args.root,)
#    elif args.catego == "SAMM":
#        save_SAMM_preprocess_weight(data_info=data,
#                                    root=args.root)
#    else:
#        save_SMIC_preprocess_weight(data_info=data,
#                                    root=args.root)
