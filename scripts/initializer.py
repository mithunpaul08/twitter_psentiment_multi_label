from argparse import Namespace
import os
import argparse
from utils.utils_rao import handle_dirs



class Initializer():
    def __init__(self):
        self._args=Namespace()

    def set_default_parameters(self):

        args = Namespace(
            # Data and Path information
            frequency_cutoff=5,
            model_state_file='model.pth',

            save_dir='saved/',
            vectorizer_file='vectorizer.json',
            glove_filepath='data/glove/glove.840B.300d.txt',


            # Training hyper parameters
            batch_size=32,
            early_stopping_criteria=5,
            seed=256,
            random_seed=20,
            weight_decay=5e-5,
            Adagrad_init=0,

            # Runtime options
            expand_filepaths_to_save_dir=True,
            reload_from_files=False,
            max_grad_norm=5,

            truncate_words_length=1000,
            embedding_size=300,
            optimizer="adagrad",
            para_init=0.01,
            hidden_sz=200,
            arch='decomp_attention',
            pretrained="false",
            update_pretrained_wordemb=False,
            cuda=True,
            workers=0,


            use_gpu=True,
            consistency_type="mse",
            NO_LABEL=-1

        )
        args.use_glove = True
        if args.expand_filepaths_to_save_dir:
            args.vectorizer_file = os.path.join(args.save_dir,
                                                args.vectorizer_file)

            args.model_state_file = os.path.join(args.save_dir,
                                                 args.model_state_file)


        self._args=args

        return args

    def create_parser(self):

        parser = argparse.ArgumentParser(description='PyTorch Mean-Teacher Training')
        parser.add_argument("--train", nargs='?', default="2018-E-c-En-train.txt")
        parser.add_argument("--dev", nargs='?', default="2018-E-c-En-dev.txt")
        parser.add_argument("--test", nargs='?', default="2018-E-c-En-test.txt")
        parser.add_argument("test_for_hacking", nargs='?', default="2018-E-c-En-dev_for_hacking.txt")
        parser.add_argument('--reload_from_files', default=False, type=self.str2bool, metavar='BOOL',
                            help='use a saved model and vectorizer')
        parser.add_argument('--num_epochs', default=100, type=int,
                            help='no of epochs to run for')
        parser.add_argument('--threshold_prediction', default=0.5, type=float,
                            help='the threshold above which the sigmoid output will be 1, else 0')
        parser.add_argument('--learning_rate', default=0.00001, type=float,
                            help='starting learning rate')
        parser.add_argument('--patience', default=5, type=float,
                            help='patience for early stopping')
        return parser

    def parse_commandline_args(self):
        return self.create_parser().parse_args(namespace=self._args)

    def str2bool(self,v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
