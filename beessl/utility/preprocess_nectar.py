import os
import zipfile
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch
import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE = 16000

train_tag = ['0006', '3629', '3631', '3640', '3690', '3693'] # ==> ~55%
valid_tag = ["3692", "3628"] # ==> ~22%
test_tag = ["3627", "3691"] # ==> ~23%

def exp_interpolate(group):
    df = pd.DataFrame()
    df["Date"] = pd.date_range(start=group.Date.min(), end=group.Date.max())
    df = df.merge(group, on="Date", how="outer")
    df["FramesOfBees"] = df["FramesOfBees"].apply(np.log).interpolate(method="linear").apply(np.exp)

    return df


def process_file(fname, output_path, step):
    try:
        waveform, sample_rate = torchaudio.load(fname)
        resampler = T.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)

        # We split in chunks each file
        num_frames = waveform.shape[1]
        chunks = num_frames // step
        chunked_waveform = torch.chunk(waveform, chunks, dim=1)
        for idx in range(chunks-1):
            id_ = os.path.basename(fname).split(".")[0] + f"___chunk_{idx}"

            new_wav_file = os.path.join(output_path, id_ + ".wav")
            torchaudio.save(new_wav_file, chunked_waveform[idx], SAMPLE_RATE)
    except:
        print(f'Error processing {fname}')

    finally:
        os.remove(fname)


def main(input_path, output_path, step_size):
    # Preprocess the original label dataset
    labels_2021 = pd.read_excel(os.path.join(input_path, '2021', 'rooftop_hives_labels_2021.xlsx'))
    labels_2021 = labels_2021[["Date", "Tag", "Fob 1st", "Fob 2nd", "Fob 3rd"]]
    labels_2021 = labels_2021.dropna(subset="Date")
    labels_2021["Date"] = pd.to_datetime(labels_2021["Date"], format="%d-%m-%Y")

    labels_2021 = labels_2021.fillna(0)
    labels_2021["Tag"] = labels_2021.Tag.astype(str)
    labels_2021["Tag"] = labels_2021["Tag"].str.replace("\xa0", "")
    labels_2021["Tag"] = labels_2021["Tag"].str.replace("2000006", "0006")
    labels_2021["FramesOfBees"] = labels_2021[["Fob 1st", "Fob 2nd", "Fob 3rd"]].sum(axis=1)
    labels_2021 = labels_2021[labels_2021["FramesOfBees"] > 0]
    labels_2021 = labels_2021.drop(["Fob 1st", "Fob 2nd", "Fob 3rd"], axis=1)
    labels_2021 = labels_2021.groupby("Tag")[["Date", "FramesOfBees"]].apply(lambda g: exp_interpolate(g)).reset_index()

    # Get the available zip files
    fnames_list = glob(f'{input_path}/**/*.zip', recursive=True)
    df = pd.DataFrame({'zip': fnames_list})
    df["Date"] = df["zip"].str.extract(r'(\d{2}-\d{2}-\d{4})')
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    df["Datetime"] = df["zip"].str.extract(r'(\d{2}-\d{2}-\d{4}_\d{2}h\d{2})')
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%m-%Y_%Hh%M")

    df["Tag"] = df["zip"].apply(lambda x: x.split("-")[-1][:-4])

    # Merge both datasets
    merged = pd.merge(labels_2021, df, on=["Date", "Tag"], how="inner").sort_values(by=["Tag", "Datetime"])
    merged = merged.set_index(merged["Datetime"], drop=True)
    merged = merged[["zip", "Tag", "FramesOfBees"]]

    merged = merged.between_time(start_time='22:00:00',end_time='06:00:00')
    merged = merged.groupby("Tag").sample(80, random_state=42)

    # First, extract all zip files
    for fname in tqdm(merged.zip):
        try:
            with zipfile.ZipFile(fname, 'r') as zip_ref:
                zip_ref.extractall(output_path)
        except:
            print(f'Error extracting {fname}')
            continue

    # Next, resample all files to 16 kHz and split in chunks
    fnames_list = glob(f'{output_path}/*.wav')
    for fname in tqdm(fnames_list):
        process_file(fname, output_path, step_size)

    # Save label file to disk
    merged["split"] = None
    merged.loc[merged.Tag.isin(train_tag), "split"] = "train"
    merged.loc[merged.Tag.isin(valid_tag), "split"] = "valid"
    merged.loc[merged.Tag.isin(test_tag), "split"] = "test"

    fnames_list = glob(f'{output_path}/*.wav')
    fnames_list = [os.path.basename(fname) for fname in fnames_list]
    df_output_path = pd.DataFrame({"wav": fnames_list})

    df_output_path["base_name"] = df_output_path.wav.str.split("__").str[0]
    merged["base_name"] = [os.path.basename(fname)[:-4] for fname in merged.zip]

    final_df = pd.merge(df_output_path, merged, on=["base_name"], how="inner")
    final_df[["wav", "FramesOfBees", "split"]].to_csv(
        os.path.join(output_path, "labels.csv"), index=False
    )


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str, default='raw_files')
    argparser.add_argument('--output_path', type=str, default='processed_vF')
    argparser.add_argument('--step_size', type=int, default=240000)
    args = argparser.parse_args()

    main(args.input_path, args.output_path, args.step_size)