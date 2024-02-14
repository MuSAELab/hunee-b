import os
import json
import random
import logging
import warnings
from tqdm import tqdm
from typing import Tuple
from audiomentations import Compose
from audiomentations import TimeMask
from audiomentations import BandPassFilter
from audiomentations import AddGaussianSNR
from audiomentations import SpecFrequencyMask

import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import get_all_files

logger = logging.getLogger(__name__)

def prepare_nectar(
    data_folder, save_folder, skip_prep=False, extension=[".wav", ".WAV"], chunkwise=False
):
    if skip_prep:
        return

    # Setting ouput files
    save_json_train = os.path.join(save_folder, "train.json")
    save_json_valid = os.path.join(save_folder, "valid.json")

    if skip(save_json_train):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    wav_lst_train = get_all_files(data_folder, match_or=extension)
    # wav_lst_train = wav_lst_train[:1000] #@todo: remove this line

    # Split the data into train and valid
    random.shuffle(wav_lst_train)
    n_samples = int(len(wav_lst_train)*0.9)
    wav_lst_valid = wav_lst_train[n_samples:]
    wav_lst_train = wav_lst_train[:n_samples]

    # Creating the json files
    create_json(wav_lst_train, save_json_train)
    create_json(wav_lst_valid, save_json_valid)


def create_json(wav_lst, json_file):
    logger.debug(f"Creating json lists in {json_file}")

    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in tqdm(wav_lst):
        id_ = os.path.basename(wav_file).split(".")[0]
        #@TODO: Get sample_rate from hparams
        duration = torchaudio.info(wav_file).num_frames / 16000
        json_dict[id_] = {
            "wav": wav_file,
            "duration": duration
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
    @sb.utils.data_pipeline.takes("wav", "duration")
    @sb.utils.data_pipeline.provides("bee_sig")
    def wav_pipeline(wav, duration):
        # Randomly crop the wav files
        duration = duration * hparams["sample_rate"]
        start = torch.randint(0, int(duration - hparams["sig_length"]), (1,))
        wav = sb.dataio.dataio.read_audio({
            "file": wav,
            "start": start,
            "stop": start + hparams["sig_length"],
        }).squeeze().numpy()

        return wav

    # Define datasets
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_root"]},
            dynamic_items=[wav_pipeline],
            output_keys=["id", "bee_sig"],
        )

    return datasets


class TrainTransform:
    def __init__(self, sample_rate:int=16000):
        self.sample_rate = sample_rate
        self.transform = Compose([
            AddGaussianSNR(min_snr_db=10.0, max_snr_db=20.0, p=0.5),
            TimeMask(min_band_part=0.1, max_band_part=0.2, fade=True, p=0.5),
            BandPassFilter(
                min_center_freq=300.0,
                max_center_freq=1000.0,
                min_bandwidth_fraction=0.8,
                max_bandwidth_fraction=1.5,
                p=0.3
            ),
        ])

        self.transform_prime = Compose([
            AddGaussianSNR(min_snr_db=10.0, max_snr_db=20.0, p=0.5),
            TimeMask(min_band_part=0.1, max_band_part=0.2, fade=True, p=0.5),
            BandPassFilter(
                min_center_freq=300.0,
                max_center_freq=1000.0,
                min_bandwidth_fraction=0.8,
                max_bandwidth_fraction=1.5,
                p=0.3
            ),
        ])

    def __call__(self, samples) -> Tuple[torch.Tensor, torch.Tensor]:
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        device = samples.device
        x, x_prime = [], []
        for audio in torch.unbind(samples, dim=0):
            audio = audio.squeeze().detach().cpu().numpy()
            x1 = self.transform(audio, sample_rate=self.sample_rate)
            x2 = self.transform_prime(audio, sample_rate=self.sample_rate)

            # Save back to torch tensor
            x.append(torch.tensor(x1).unsqueeze(0))
            x_prime.append(torch.tensor(x2).unsqueeze(0))

        return torch.cat(x).to(device), torch.cat(x_prime).to(device)
