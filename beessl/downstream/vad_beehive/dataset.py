import os
import json
import logging
import torch
import torchaudio
import speechbrain as sb
from speechbrain.utils.data_utils import get_all_files
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def prepare_nuhive(data_folder, win_length=0.25, sample_rate=16000):
    # Setting input files
    train_files = [
        os.path.join(data_folder, "Hive3_28_07_2017_QueenBee____07_50_00.wav"),
        os.path.join(data_folder, "Hive3_28_07_2017_QueenBee____08_00_00.wav"),
    ]

    test_files = [
        os.path.join(data_folder, "Hive3_28_07_2017_QueenBee____08_10_00.wav"),
        os.path.join(data_folder, "Hive3_28_07_2017_QueenBee____08_20_00.wav"),
    ]

    training_set = [parse_file(t, win_length=win_length, sample_rate=sample_rate) for t in train_files]
    testing_set = [parse_file(t, win_length=win_length, sample_rate=sample_rate) for t in test_files]

    X_train, y_train = zip(*training_set)
    X_test, y_test = zip(*testing_set)
    X_train = torch.cat(X_train, dim=0)
    X_test = torch.cat(X_test, dim=0)
    y_train = torch.cat(y_train, dim=0)
    y_test = torch.cat(y_test, dim=0)

    X_train, X_val, y_train, y_val  = train_test_split(
        X_train, y_train,
        stratify=y_train,
        random_state=42,
    )

    return {
        "train": torch.utils.data.TensorDataset(X_train, y_train),
        "valid": torch.utils.data.TensorDataset(X_val, y_val),
        "test": torch.utils.data.TensorDataset(X_test, y_test),
    }


def parse_file(path, win_length=0.25, sample_rate=16000):
        fname = os.path.basename(path)
        x, sr = torchaudio.load(path)
        

        label = torch.zeros_like(x)
        step = int(win_length*sample_rate)
        length = x.shape[1]

        if groundtruth_vad_nuhive[fname] is not None:
            for start, stop in groundtruth_vad_nuhive[fname]:
                label[:, (start*sample_rate):(stop*sample_rate)] = 1

        label = torch.split(label, step, dim=1)[:-1]
        X_train = torch.split(x, step, dim=1)[:-1]
        X_train = torch.stack(X_train, dim=0)

        y_train = torch.stack([torch.round(y.sum()/step) for y in label], dim=0)

        return X_train.squeeze(1), y_train.unsqueeze(-1)


# Groundtruth VAD for the QueenBee dataset
groundtruth_vad_nuhive = {
    'Hive3_28_07_2017_QueenBee____07_50_00.wav':[
        [185, 186],
        [205, 206],
        [214, 215],
        [225, 226],
        [232, 233],
        [236, 237],
        [240, 241],
        [249, 251],
        [254, 257],
        [259, 263],
        [274, 275],
        [281, 282],
        [284, 285],
        [289, 291],
        [293, 295],
        [298, 310],
        [312, 313],
        [316, 317],
        [320, 321],
        [324, 327],
        [330, 331],
        [335, 336],
        [345, 347],
        [354, 357],
        [359, 360],
        [364, 367],
        [369, 372],
        [376, 377],
        [383, 392],
        [397, 398],
        [401, 403],
        [405, 406],
        [415, 416],
        [421, 426],
        [429, 430],
        [433, 434],
        [436, 444],
        [451, 453],
        [457, 458],
        [459, 463],
        [465, 466],
        [471, 473],
        [484, 485],
        [493, 497],
        [499, 501],
        [504, 505],
        [507, 509],
        [511, 514],
        [517, 518],
        [534, 535],
        [553, 554],
        [557, 559],
        [560, 561],
        [563, 565],
        [567, 568],
        [572, 574],
        [576, 578],
        [582, 584],
        [592, 594],
        [596, 597]
    ],
    
    'Hive3_28_07_2017_QueenBee____08_00_00.wav':[
        [7, 17],
        [22, 24],
        [29, 30],
        [47, 49],
        [52, 53],
        [56, 62],
        [63, 64],
        [77, 83],
        [85, 86],
        [117, 120],
        [122, 123],
        [125, 133],
        [139, 140],
        [156, 169],
        [186, 187],
        [209, 210],
        [219, 223],
        [227, 228],
        [239, 240],
        [250, 254],
        [255, 256],
        [258, 259],
        [265, 266],
        [270, 273],
        [275, 278],
        [287, 288],
        [292, 293],
        [305, 306],
        [314, 315],
        [317, 318],
        [319, 321],
        [326, 328],
        [330, 331],
        [333, 334],
        [337, 338],
        [349, 350],
        [355, 356],
        [393, 397],
        [404, 405],
        [407, 408],
        [424, 425],
        [431, 432],
        [436, 446],
        [449, 451],
        [457, 459],
        [462, 463],
        [468, 469],
        [471, 472],
        [478, 483],
        [487, 488],
        [497, 512],
        [516, 519],
        [522, 531],
        [533, 536],
        [546, 547]
    ],

    'Hive3_28_07_2017_QueenBee____08_10_00.wav':[
          [3,5],
          [8,11],
          [14, 15],
          [20,21],
          [28, 29],
          [30, 31],
          [40, 41],
          [45, 46],
          [51, 53],
          [58, 66],
          [69, 70],
          [80, 81],
          [89, 91],
          [102, 103],
          [106, 107],
          [155, 156],
          [157, 158],
          [159, 160],
          [188, 189],
          [192, 193],
          [198, 199],
          [209, 210],
          [294, 295],
          [308, 309],
          [319, 320],
          [327, 328],
          [338, 339],
          [364, 365],
          [440, 441]
      ],

      'Hive3_28_07_2017_QueenBee____08_20_00.wav':[
            [107, 108],
            [156, 157],
            [226, 227],
            [264, 265],
            [267, 268],
            [273, 274],
            [298, 299],
            [339, 340],
            [346, 347],
            [352, 353],
            [357, 358],
            [374, 375],
            [389, 390],
            [415, 416],
            [418, 419],
            [423, 424],
            [739, 740],
            [528, 529],
            [579, 580]
        ],

        'Hive3_28_07_2017_QueenBee____08_30_00.wav':[
            [5, 6],
            [107, 108],
            [118, 119],
            [157, 158],
            [183, 184],
        ]
}