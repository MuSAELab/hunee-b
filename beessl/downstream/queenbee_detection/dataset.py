import os
import json
import logging
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import get_all_files

logger = logging.getLogger(__name__)

def prepare_nuhive(
    data_folder, save_folder, skip_prep=False, extension=".wav", chunkwise=False
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

    train_folder = os.path.join(data_folder, "train")
    valid_folder = os.path.join(data_folder, "valid")
    test_folder = os.path.join(data_folder, "test")

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    wav_lst_train = get_all_files(train_folder, match_and=extension)
    wav_lst_valid = get_all_files(valid_folder, match_and=extension)
    wav_lst_test = get_all_files(test_folder, match_and=extension)

    for params in [
        (wav_lst_train, save_json_train),
        (wav_lst_valid, save_json_valid),
        (wav_lst_test, save_json_test)
    ]:
        create_json(*params, chunkwise=chunkwise)


def create_json(wav_lst, json_file, chunkwise=False):
    logger.debug(f"Creating json lists in {json_file}")

    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_lst:
        id_ = os.path.basename(wav_file).split(".")[0]
        label = 0 if "NO_QueenBee" in id_ else 1
        duration = torchaudio.info(wav_file).num_frames / 16000
        json_dict[id_] = {
            "wav": wav_file,
            "duration": duration,
            "label": label,
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
    # Define audio pipelines
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("bee_sig")
    def wav_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)
    
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("target")
    def label_pipeline(label):
        return torch.tensor([float(label)])

    # Define datasets
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_root"]},
            dynamic_items=[wav_pipeline, label_pipeline],
            output_keys=["id", "bee_sig", "target"],
        )

    return datasets