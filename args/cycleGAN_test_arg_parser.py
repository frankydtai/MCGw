"""
Arguments for MaskCycleGAN-VC testing.
Inherits BaseArgParser.
"""

from args.base_arg_parser import BaseArgParser

class CycleGANTestArgParser(BaseArgParser):
    """
    Class which implements an argument parser for args used only in training MaskCycleGAN-VC.
    It inherits TrainArgParser.
    """

    def __init__(self):
        super(CycleGANTestArgParser, self).__init__()
        self.parser.add_argument('--sample_rate', type=int, default=22050, help='Sampling rate of mel-spectrograms.')
        self.parser.add_argument(
            '--speaker_A_id', type=str, default="VCC2SF3", help='Source speaker id (From VOC dataset).')
        self.parser.add_argument(
            '--speaker_B_id', type=str, default="VCC2TF1", help='Source speaker id (From VOC dataset).')
        self.parser.add_argument(
            '--preprocessed_data_dir', type=str, default="vcc2018_training_preprocessed/", help='Directory containing preprocessed dataset files.')
        self.parser.add_argument(
            '--ckpt_dir', type=str, default=None, help='Path to model ckpt.')
        self.parser.add_argument(
            '--model_name', type=str, choices=('generator_A2B', 'generator_B2A'), default='generator_A2B', help='Name of model to load.')
        self.parser.add_argument(
            "--converted_audio_subdir", type=str, default="converted_audio", help="Subfolder name where converted audio files will be saved")
        #new
        self.parser.add_argument("--vocoder", type=str, default="hifi", choices=("mel", "hifi"), help="Which vocoder to use: 'mel' for MelGAN, 'hifi' for HiFi-GAN")