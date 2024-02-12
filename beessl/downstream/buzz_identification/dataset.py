import os
import json
import logging
import pandas as pd

import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import get_all_files

logger = logging.getLogger(__name__)

def prepare_buzz(
    data_folder, save_folder, skip_prep=False, extension=".wav"
):
    if skip_prep:
        return
    
    # Setting ouput files
    save_json_train = os.path.join(save_folder, "train.json")
    save_json_valid = os.path.join(save_folder, "valid.json")
    save_json_test = os.path.join(save_folder, "test.json")

    if skip(save_json_train, save_json_test, save_json_valid):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    buzz_df = pd.read_csv(os.path.join(data_folder, "labels.csv"))
    wav_lst_train = buzz_df[buzz_df.split == "train"]
    wav_lst_valid = buzz_df[buzz_df.split == "valid"]
    wav_lst_test = buzz_df[buzz_df.split == "test"]

    for params in [
        (wav_lst_train, save_json_train),
        (wav_lst_valid, save_json_valid),
        (wav_lst_test, save_json_test)
    ]:
        create_json(*params, data_folder=data_folder)


def create_json(dataframe, json_file, data_folder):
    logger.debug(f"Creating json lists in {json_file}")

    # Processing all the wav files in the list
    json_dict = {}
    for index, row in dataframe.iterrows():
        id_ = os.path.basename(row["id_"]).split(".")[0]
        wav_file, label = os.path.join(data_folder, row["id_"]), row["label"]
        json_dict[id_] = {
            "wav": wav_file,
            "label": label
        }

    # Writing the json lines
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the NuHive data_preparation has been already done.
    If the preparation has been done, we can skip it.
    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def dataio_prep(hparams):
    # Define datasets
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    # Define audio pipelines
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def wav_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)
    
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    label_encoder.update_from_iterable(["bee", "noise", "cricket"])

    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("target")
    def label_pipeline(label):
        return label_encoder.encode_label(label)

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_root"]},
            dynamic_items=[wav_pipeline, label_pipeline],
            output_keys=["id", "sig", "target"],
        )
    
    lab_enc_file = os.path.join(hparams["annotation_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[datasets["train"]], output_key="target",
    )

    return datasets, label_encoder