import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchaudio

from mask_cyclegan_vc.model import Generator, Discriminator
from args.cycleGAN_test_arg_parser import CycleGANTestArgParser
from dataset.vc_dataset import VCDataset
from mask_cyclegan_vc.utils import decode_melspectrogram
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver

# — at the top of test.py —
import json            # NEW
import hifigan         # NEW
import yaml

def load_hifigan(config, checkpoint_path="./hifigan/g_00205000", device='cuda'):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]
    assert speaker == 'universal'
    assert name == "HiFi-GAN16k"

    #print("#### HiFI-GAN16k", name, speaker, device)
    with open("./hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    #print("### HiFI-GAN ckpt", checkpoint_path)
    if checkpoint_path.startswith("http"):
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.hub.load_state_dict_from_url(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu')) if device!='cuda' else torch.load(checkpoint_path)

    vocoder.load_state_dict(ckpt['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder

class MaskCycleGANVCTesting(object):
    """Tester for MaskCycleGAN-VC
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store Args
        self.device = args.device
        self.converted_audio_dir = os.path.join(args.save_dir, args.name, args.converted_audio_subdir)
        os.makedirs(self.converted_audio_dir, exist_ok=True)
        self.model_name = args.model_name

        self.speaker_A_id = args.speaker_A_id
        self.speaker_B_id = args.speaker_B_id
        # Initialize MelGAN-Vocoder used to decode Mel-spectrograms
        if args.vocoder == "mel":
            self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
        #hifi
        else:
            self.model_config = yaml.load(open("./hifigan/my_model.yaml", "r"), Loader=yaml.FullLoader)
            self.vocoder = load_hifigan(self.model_config, checkpoint_path="./hifigan/g_02500000", device=self.device).eval()
        self.sample_rate = args.sample_rate

        # Initialize speakerA's dataset
        self.dataset_A = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, self.speaker_A_id, f"{self.speaker_A_id}_normalized.pickle"))
        dataset_A_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, self.speaker_A_id, f"{self.speaker_A_id}_norm_stat.npz"))
        self.dataset_A_mean = dataset_A_norm_stats['mean']
        self.dataset_A_std = dataset_A_norm_stats['std']
        
        # Initialize speakerB's dataset
        self.dataset_B = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, self.speaker_B_id, f"{self.speaker_B_id}_normalized.pickle"))
        dataset_B_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, self.speaker_B_id, f"{self.speaker_B_id}_norm_stat.npz"))
        self.dataset_B_mean = dataset_B_norm_stats['mean']
        self.dataset_B_std = dataset_B_norm_stats['std']

        source_dataset = self.dataset_A if self.model_name == 'generator_A2B' else self.dataset_B
        self.dataset = VCDataset(datasetA=source_dataset,
                                 datasetB=None,
                                 valid=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           drop_last=False)

        # Generator
        self.generator = Generator().to(self.device)
        self.generator.eval()

        # Load Generator from ckpt
        self.saver = ModelSaver(args)
        self.saver.load_model(self.generator, self.model_name)

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def test(self):
        merged_audio = []   
        for i, sample in enumerate(tqdm(self.test_dataloader)):

            save_path = None
            if self.model_name == 'generator_A2B':
                real_A = sample
                real_A = real_A.to(self.device, dtype=torch.float)
                fake_B = self.generator(real_A, torch.ones_like(real_A))

                wav_fake_B = decode_melspectrogram(self.vocoder, fake_B[0].detach(
                ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()

                wav_real_A = decode_melspectrogram(self.vocoder, real_A[0].detach(
                ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()

                if wav_fake_B.ndim == 3:
                    wav_fake_B=wav_fake_B.squeeze(1)
                
                save_path = os.path.join(self.converted_audio_dir, f"{i}-converted_{self.speaker_A_id}_to_{self.speaker_B_id}.wav")
                save_path_orig = os.path.join(self.converted_audio_dir,
                                        f"{i}-original_{self.speaker_A_id}_to_{self.speaker_B_id}.wav")
                torchaudio.save(save_path, wav_fake_B, sample_rate=self.sample_rate)
                torchaudio.save(save_path_orig, wav_real_A, sample_rate=self.sample_rate)
                # if wav_fake_B.ndim == 3:
                #     wav_fake_B=wav_fake_B.unsqueeze(0)
                #merged_audio.append(wav_fake_B)

            else:
                real_B = sample
                real_B = real_B.to(self.device, dtype=torch.float)
                fake_A = self.generator(real_B, torch.ones_like(real_B))

                wav_fake_A = decode_melspectrogram(self.vocoder, fake_A[0].detach(
                ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()

                if wav_fake_A.ndim == 3:
                    wav_fake_A=wav_fake_A.squeeze(1)

                wav_real_B = decode_melspectrogram(self.vocoder, real_B[0].detach(
                ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
                save_path = os.path.join(self.converted_audio_dir, f"{i}-converted_{self.speaker_B_id}_to_{self.speaker_A_id}.wav")
                save_path_orig = os.path.join(self.converted_audio_dir,
                                       f"{i}-original_{self.speaker_B_id}_to_{self.speaker_A_id}.wav")
                torchaudio.save(save_path, wav_fake_A, sample_rate=self.sample_rate)
                torchaudio.save(save_path_orig, wav_real_B, sample_rate=self.sample_rate)

                # if merged_audio.ndim == 3:
                #     wav_fake_A=wav_fake_A.unsqueeze(0)
                #merged_audio.append(wav_fake_A)

            # merged_wav = torch.cat(merged_audio, dim=1)
            # # if merged_wav.ndim == 3:                              # e.g., [1, 1, T] -> [1, T]
            # #     merged_wav = merged_wav.squeeze(0)
            # # if merged_wav.ndim == 1:                              # ensure [C, T]
            # #     merged_wav = merged_wav.unsqueeze(0)
            # merged_path = os.path.join(self.converted_audio_dir, "merged.wav")
            # torchaudio.save(merged_path, merged_wav , sample_rate=self.sample_rate)

if __name__ == "__main__":
    parser = CycleGANTestArgParser()
    args = parser.parse_args()
    tester = MaskCycleGANVCTesting(args)
    tester.test()
