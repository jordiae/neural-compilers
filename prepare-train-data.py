from neural_compilers.utils.config import DataGenConfig, TokenizerConfig
from neural_compilers.train.data_generator import DataGen
import json


def main():
    import argparse
    parser = argparse.ArgumentParser('Train data generator')
    parser.add_argument('--config-file', help='Path to config file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    config = DataGenConfig.from_dict(config)
    config.config_path = args.config_file
    data_gen = DataGen(config)
    data_gen.generate()


if __name__ == '__main__':
    main()
