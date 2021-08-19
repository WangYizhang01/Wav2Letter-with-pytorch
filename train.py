"""
@author: Wang Yizhang <1739601638@qq.com>
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from Wav2Letter.model import Wav2Letter
from Wav2Letter.data import GoogleSpeechCommand
from Wav2Letter.decoder import GreedyDecoder
import os
import math


def train(batch_size, epochs):
    # load saved numpy arrays for google speech command
    gs = GoogleSpeechCommand()
    _inputs, _targets = gs.load_vectors("./speech_data")

    # paramters
    mfcc_features = 13
    grapheme_count = gs.intencode.grapheme_count

    print("training google speech dataset")
    print("data size", len(_inputs))
    print("batch_size", batch_size)
    print("epochs", epochs)
    print("num_mfcc_features", mfcc_features)
    print("grapheme_count", grapheme_count)

    # torch tensors
    inputs = torch.Tensor(_inputs).cuda()
    targets = torch.IntTensor(_targets).cuda()

    # split train, eval
    data_size = len(_inputs)
    train_inputs = inputs[0:int(0.9*data_size)]
    train_targets = targets[0:int(0.9*data_size)]
    eval_inputs = inputs[int(0.9*data_size):-1000]
    eval_targets = targets[int(0.9*data_size):-1000]
    # print("train_inputs.size() ", train_inputs.size())
    # exit(0)

    # Initialize model, loss, optimizer
    model = Wav2Letter(mfcc_features, grapheme_count)
    model.cuda()
    # print(model.layers)

    ctc_loss = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    train_total_steps = math.ceil(len(train_inputs) / batch_size)
    eval_total_steps = math.ceil(len(eval_inputs) / batch_size)

    for epoch in range(epochs):

        samples_processed = 0
        avg_epoch_loss = 0

        for step in range(train_total_steps):
            optimizer.zero_grad()
            train_data_batch = train_inputs[samples_processed : batch_size + samples_processed].transpose(1, 2)

            log_probs = model(train_data_batch)
            log_probs = log_probs.transpose(1, 2).transpose(0, 1)

            mini_batch_size = len(train_data_batch)
            targets = train_targets[samples_processed: mini_batch_size + samples_processed]

            input_lengths = torch.full((mini_batch_size,), log_probs.shape[0], dtype=torch.long)
            target_lengths = torch.IntTensor([target.shape[0] for target in targets])

            # print("log_probs", log_probs.size(), log_probs)
            # print("targets", targets.size(), targets)
            # print("input_lengths", input_lengths.size(), input_lengths.dtype,input_lengths)
            # print("target_lengths", target_lengths.size(), target_lengths.dtype, train_targets)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            # print('loss', loss)
            # exit(0)
            avg_epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            samples_processed += mini_batch_size

        # evaluate
        decoder = GreedyDecoder()
        wer = 0
        start_index = 0
        for step in range(eval_total_steps):
            eval_data_batch = eval_inputs[start_index : batch_size + start_index].transpose(1, 2)
            min_batch_size = len(eval_data_batch)
            eval_targets_batch = eval_targets[start_index : min_batch_size + start_index]
            eval_log_props = model(eval_data_batch)

            output = decoder.decode(eval_log_props)
            pred_strings, output = decoder.convert_to_strings(output)
            eval_target_strings = decoder.convert_to_strings(eval_targets_batch, remove_repetitions=False, return_offsets=False)
            wer += decoder.wer(eval_target_strings, pred_strings)
            start_index += min_batch_size
        
        print("epoch", epoch + 1, "average epoch loss", avg_epoch_loss / train_total_steps, "wer", wer/eval_total_steps)
        if (epoch + 1) % 100 == 0:
            torch.save(model, os.path.join('save_models', "model_{}.pth".format(epoch+1)))


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
    train(batch_size, epochs)
