import os
import argparse
import pandas as pd
from glob import glob
from tqdm.auto import tqdm

import torchaudio
import torchaudio.transforms as T

SAMPLE_RATE = 16000

def process_file(row, output_path):
    try:
        waveform, sample_rate = torchaudio.load(row.path)
        resampler = T.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)
        torchaudio.save(os.path.join(output_path, row.id_), waveform, SAMPLE_RATE)

        return len(waveform[0]) / SAMPLE_RATE
    except:
        print(f'Error processing {row.id_}')


def main(input_path, output_path):
    input_path = os.path.join(input_path, "**/*.wav")
    all_files = glob(input_path, recursive=True)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.DataFrame({"path": all_files})
    df["split"] = df.path.apply(lambda v: v.split("/")[1])
    df["split"] = df.split.str.replace("test", "valid")
    df["split"] = df.split.str.replace("out_of_sample_data_for_validation", "test")

    df["id_"] = df.path.apply(lambda v: v.split("/")[-1])
    df["label"] = df.path.apply(lambda v: v.split("/")[2].split("_")[0])

    tqdm.pandas(desc="Resampling the data")
    df["length"] = df.progress_apply(lambda r: process_file(r, output_path), axis=1)

    csv_path = os.path.join(output_path, "labels.csv")
    df[["id_", "length", "split", "label"]].to_csv(csv_path, index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_path', type=str, default='BUZZ3')
    argparser.add_argument('--output_path', type=str, default='processed')
    args = argparser.parse_args()

    main(args.input_path, args.output_path)
