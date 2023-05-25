import torch.nn as nn
import torch
import numpy as np
from utils import hidden_encoder_conversion


class Encoder(nn.Module):
    def __init__(self, tokens_num, embedding_matrix, num_layers, hidden_size, embedding_dropout, rnn_dropout):
        super().__init__()
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.num_layers = num_layers
        self.forward_reverse_hidden_size = hidden_size * 2

        self.embedding = nn.Embedding(
            num_embeddings=tokens_num,
            embedding_dim=embedding_matrix.shape[1],
        ).requires_grad_(False)

        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=rnn_dropout
        )

    def forward(self, x):
        # x.size() = [batch_size, seq_length]

        x = self.embedding_dropout(self.embedding(x))

        # x.size() = [batch_size, seq_length, embedding_dim]

        x, (hidden, cell) = self.lstm(x)

        # x.size() = [batch_size, seq_length, n_directions * hidden_size]
        # hidden.size() = [n_layers * n_directions, batch_size, hidden_size]
        # cell.size() = [n_layers * n_directions, batch_size, hidden_size]

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, tokens_num, embedding_matrix, embedding_dropout, main_hidden_size, num_layers, rnn_dropout):
        super().__init__()
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.tokens_num = tokens_num
        self.num_layers = num_layers
        self.hidden_size = main_hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=tokens_num,
            embedding_dim=embedding_matrix.shape[1],
        ).requires_grad_(False)

        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=main_hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=rnn_dropout
        )

        self.out_linear = nn.Linear(
            in_features=main_hidden_size,
            out_features=tokens_num
        )

    def forward(self, x, hidden, cell):
        # x.size() = [batch_size]
        # hidden.size() = [decoder_n_layers, decoder_batch_size, decoder_hidden_size * decoder_n_directions]
        # cell.size() = [decoder_n_layers, decoder_batch_size, decoder_hidden_size * decoder_n_directions]

        x = self.embedding_dropout(self.embedding(x))

        # x.size() = [batch_size, embedding_dim]

        x = x.reshape(x.size(0), 1, x.size(1))

        # x.size() = [batch_size, seq_length=1, embedding_dim]

        x, (hidden, cell) = self.lstm(x, (hidden, cell))

        # x.size() = [batch_size, seq_length=1, n_directions * hidden_size]

        x = self.out_linear(x.squeeze(1))

        # x.size() = [batch_size, output_dim]

        return x, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, dropout_hidden, dropout_cell):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.num_layers == decoder.num_layers, \
            "Decoder and Encoder must have the same number of layers"

        self.hidden_state_mappings = nn.ModuleList()
        self.cell_state_mappings = nn.ModuleList()

        for _ in range(self.decoder.num_layers):
            hidden_state_mapping = nn.Linear(self.encoder.forward_reverse_hidden_size,
                                             self.decoder.hidden_size).to(device)
            cell_state_mapping = nn.Linear(self.encoder.forward_reverse_hidden_size,
                                           self.decoder.hidden_size).to(device)

            self.hidden_state_mappings.append(hidden_state_mapping)
            self.cell_state_mappings.append(cell_state_mapping)

        self.dropout_hidden_state = nn.Dropout(dropout_hidden)
        self.dropout_cell_state = nn.Dropout(dropout_cell)

    def forward(self, source, target, teacher_forcing_ration=0.5):
        # source.size() = [batch_size, source_seq_length]
        # target.size() = [batch_size, target_seq_length]

        batch_size, max_len = target.size()
        target_vocab_size = self.decoder.tokens_num

        outputs = torch.zeros(max_len, batch_size, target_vocab_size, device=self.device)

        hidden_encoder, cell_encoder = self.encoder(source)
        hidden_encoder, cell_encoder = (hidden_encoder_conversion(hidden_encoder, device=self.device),
                                        hidden_encoder_conversion(cell_encoder, device=self.device))

        hidden = torch.zeros(hidden_encoder.size(0), hidden_encoder.size(1), self.decoder.hidden_size, device=self.device)
        cell = torch.zeros(hidden_encoder.size(0), hidden_encoder.size(1), self.decoder.hidden_size, device=self.device)

        for idx in range(len(hidden_encoder)):
            hidden[idx] = self.dropout_hidden_state(self.hidden_state_mappings[idx](hidden_encoder[idx]))
            cell[idx] = self.dropout_cell_state(self.cell_state_mappings[idx](cell_encoder[idx]))

        # hidden_encoder = self.dropout_hidden_state(self.hidden_state_mapping(hidden_encoder))
        # cell_encoder = self.dropout_cell_state(self.cell_state_mapping(cell_encoder))

        cur_input = target[:, 0]

        for idx in range(1, max_len):

            output, hidden, cell = self.decoder(cur_input, hidden, cell)
            #  output.size() = [batch_size, output_dim]

            outputs[idx] = output

            teacher_force_flag = np.random.rand() < teacher_forcing_ration
            top1 = output.max(dim=1)[1]

            cur_input = (target[:, idx] if teacher_force_flag else top1)

        return outputs