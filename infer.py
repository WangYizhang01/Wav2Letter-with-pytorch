"""
@author: Wang Yizhang <1739601638@qq.com>
"""

import argparse
import torch
from Wav2Letter.model import Wav2Letter
from Wav2Letter.data import GoogleSpeechCommand
from Wav2Letter.decoder import GreedyDecoder
import os


def infer(batch_size):
    # load saved numpy arrays for google speech command
    gs = GoogleSpeechCommand()
    _inputs, _targets = gs.load_vectors("./speech_data")

    # paramters
    batch_size = batch_size
    mfcc_features = 13
    grapheme_count = gs.intencode.grapheme_count

    # torch tensors
    inputs = torch.Tensor(_inputs).cuda()
    targets = torch.IntTensor(_targets).cuda()

    # Initialize model, loss, optimizer
    model = Wav2Letter(mfcc_features, grapheme_count)
    model.cuda()
    model = torch.load(os.path.join('save_models', "model_1000.pth"))

    decoder = GreedyDecoder()

    inputs = inputs.transpose(1, 2)

    sample = inputs[-1000:]
    sample_target = targets[-1000:]
    
    log_probs = model(sample)
    output = decoder.decode(log_probs)

    pred_strings, output = decoder.convert_to_strings(output)
    sample_target_strings = decoder.convert_to_strings(sample_target, remove_repetitions=False, return_offsets=False)
    wer = decoder.wer(sample_target_strings, pred_strings)

    print("wer", wer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wav2Letter')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='total epochs (default: 100)')
    parser.add_argument('--save_models', type=str, default='save_models')

    args = parser.parse_args()
    
    batch_size = args.batch_size
    epochs = args.epochs
    infer(batch_size)
