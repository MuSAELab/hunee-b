import os
import logging
import argparse
import torchaudio
from tqdm import tqdm


def resample_wav(input_dir:str, output_dir:str = None, target_sr:int = 16000):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define a closure to resample a single WAV file
    def resample(wav_file, input_dir, output_dir):
        # Load the audio file using Torchaudio
        waveform, sample_rate = torchaudio.load(
            os.path.join(input_dir, wav_file)
        )

        # Resample the audio to 16 kHz using Torchaudio
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        resampled_waveform = resampler(waveform)

        # Save the resampled audio to the output directory
        output_path = os.path.join(output_dir, wav_file)
        torchaudio.save(output_path, resampled_waveform, target_sr)

    for set_ in ['train', 'test', 'valid']:
        input_split = os.path.join(input_dir, set_)
        output_split = os.path.join(output_dir, set_)

        # Create the output directory if it does not exist
        if not os.path.exists(output_split):
            os.makedirs(output_split)
        else:
            MSG = f"[SKIPPING] The folder {set_} already exists inside the output_dir"
            logging.warning(MSG)
            continue

        # Get a list of all the WAV files in the input directory
        wav_files = [f for f in os.listdir(input_split) if f.endswith('.wav')]

        # Process the WAV files in sequential mode
        for wav_file in tqdm(wav_files):
            resample(wav_file, input_split, output_split)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--target_sr', type=int, default=16000, help='Target sampling rate')
    args = parser.parse_args()

    # Resample the audio files
    resample_wav(args.input_dir, args.output_dir, args.target_sr)