from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Wav2Letter.decoder import GreedyDecoder
import os


class Wav2Letter(nn.Module):

    def __init__(self, num_features, num_classes):
        super(Wav2Letter, self).__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride)
        self.layers = nn.Sequential(
            nn.Conv1d(num_features, 250, 48, 2),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 2000, 32),
            torch.nn.ReLU(),
            nn.Conv1d(2000, 2000, 1),
            torch.nn.ReLU(),
            nn.Conv1d(2000, num_classes, 1),
        )

    def forward(self, data_batch):
        """Forward pass through Wav2Letter network than 
            takes log probability of output

        Args:
            data_batch (int): mini batch of data
             shape (batch, num_features, frame_len)

        Returns:
            log_probs (torch.Tensor):
                shape  (batch_size, num_classes, output_len)
        """
        # y_pred shape (batch_size, num_classes, output_len)
        y_pred = self.layers(data_batch)

        # compute log softmax probability on graphemes
        log_probs = F.log_softmax(y_pred, dim=1)

        return log_probs

    # def train(self, train_inputs, train_targets, eval_inputs, eval_targets, optimizer, ctc_loss, batch_size, epoch, print_every=50):
    #     """Trains Wav2Letter model.

    #     Args:
    #         train_inputs (torch.Tensor): shape (sample_size, num_features, frame_len)
    #         train_targets (torch.Tensor): shape (sample_size, seq_len)
    #         optimizer (nn.optim): pytorch optimizer
    #         ctc_loss (ctc_loss_fn): ctc loss function
    #         batch_size (int): size of mini batches
    #         epoch (int): number of epochs
    #         print_every (int): every number of steps to print loss
    #     """

    #     train_total_steps = math.ceil(len(train_inputs) / batch_size)
    #     eval_total_steps = math.ceil(len(eval_inputs) / batch_size)

    #     for t in range(epoch):

    #         samples_processed = 0
    #         avg_epoch_loss = 0

    #         for step in range(train_total_steps):
    #             optimizer.zero_grad()
    #             train_data_batch = train_inputs[samples_processed : batch_size + samples_processed].transpose(1, 2)

    #             log_probs = self.forward(train_data_batch)

    #             log_probs = log_probs.transpose(1, 2).transpose(0, 1)

    #             mini_batch_size = len(train_data_batch)
    #             targets = train_targets[samples_processed: mini_batch_size + samples_processed]

    #             input_lengths = torch.full((mini_batch_size,), log_probs.shape[0], dtype=torch.long)
    #             target_lengths = torch.IntTensor([target.shape[0] for target in targets])

    #             loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

    #             avg_epoch_loss += loss.item()

    #             loss.backward()
    #             optimizer.step()

    #             samples_processed += mini_batch_size

    #         # evaluate
    #         decoder = GreedyDecoder()
    #         wer = 0
    #         start_index = 0
    #         for step in range(eval_total_steps):
    #             eval_data_batch = eval_inputs[start_index : batch_size + start_index].transpose(1, 2)
    #             min_batch_size = len(eval_data_batch)
    #             eval_targets_batch = eval_targets[start_index : min_batch_size + start_index]
    #             eval_log_props = self.forward(eval_data_batch)

    #             output = decoder.decode(eval_log_props)
    #             pred_strings, output = decoder.convert_to_strings(output)
    #             eval_target_strings = decoder.convert_to_strings(eval_targets_batch, remove_repetitions=False, return_offsets=False)
    #             wer += decoder.wer(eval_target_strings, pred_strings)
    #             start_index += min_batch_size
            
    #         print("epoch", t + 1, "average epoch loss", avg_epoch_loss / train_total_steps, "wer", wer/eval_total_steps)


    # def eval(self, sample):
    #     """Evaluate model given a single sample

    #     Args:
    #         sample (torch.Tensor): shape (n_features, frame_len)

    #     Returns:
    #         log probabilities (torch.Tensor):
    #             shape (n_features, output_len)
    #     """
    #     # _input = sample.reshape(batch_size, sample.shape[0], sample.shape[1])
    #     _input = sample
    #     log_prob = self.forward(_input)
    #     return log_prob
