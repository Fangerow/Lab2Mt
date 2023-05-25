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
        )

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

        return x, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_linear = nn.Linear(in_features=hidden_size * 2, out_features=1)
        self.attention_softmax = nn.Softmax(dim=1)

    def forward(self, tens):
        weights = self.attention_softmax(self.attention_linear(tens))
        weights = weights.permute(0, 2, 1)
        return weights


class Decoder(nn.Module):
    def __init__(self, tokens_num, embedding_matrix, embedding_dropout, main_hidden_size, num_layers, rnn_dropout):
        super().__init__()
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.tokens_num = tokens_num
        self.num_layers = num_layers
        self.hidden_size = main_hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=tokens_num,
            embedding_dim=embedding_matrix.shape[1]
        )

        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=main_hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=rnn_dropout
        )

        self.attention = Attention(main_hidden_size)

        self.out_linear = nn.Linear(
            in_features=main_hidden_size * 2,
            out_features=tokens_num
        )

    def forward(self, x, hidden, cell, x_encoder, device):
        # x.size() = [batch_size]
        # hidden.size() = [decoder_n_layers, decoder_batch_size, decoder_hidden_size * decoder_n_directions]
        # cell.size() = [decoder_n_layers, decoder_batch_size, decoder_hidden_size * decoder_n_directions]

        x = self.embedding_dropout(self.embedding(x))

        # x.size() = [batch_size, embedding_dim]

        x = x.reshape(x.size(0), 1, x.size(1))

        # x.size() = [batch_size, seq_length=1, embedding_dim]

        x, (hidden, cell) = self.lstm(x, (hidden, cell))

        x_new = torch.zeros([x.size(0), x_encoder.size(1), x.size(2)]).to(device)

        x_new = x.repeat(1, 1, x_new.size(1)).view(x_new.size(0), x_new.size(1), -1)

        tens = torch.cat([x_encoder, x_new], dim=2)

        weights = self.attention.forward(tens)

        #         weighted_tensor = x_new * weights[:, None]
        weighted_tensor = torch.matmul(weights, x_new).squeeze(1)

        # result = weighted_tensor.sum(dim=0)

        # x.size() = [batch_size, seq_length=1, n_directions * hidden_size]
        x = torch.cat([x.squeeze(1), weighted_tensor], dim=1)

        x = self.out_linear(x)

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

        self.hidden_state_mappings = nn.ModuleList().to(self.device)
        self.cell_state_mappings = nn.ModuleList().to(self.device)

        for _ in range(self.decoder.num_layers):
            self.hidden_state_mappings.append(
                nn.Linear(self.encoder.forward_reverse_hidden_size, self.decoder.hidden_size // 2))
            self.cell_state_mappings.append(
                nn.Linear(self.encoder.forward_reverse_hidden_size, self.decoder.hidden_size // 2))

        self.dropout_hidden_state = nn.Dropout(dropout_hidden)
        self.dropout_cell_state = nn.Dropout(dropout_cell)

    def forward(self, source, target, teacher_forcing_ration=0.5):
        # source.size() = [batch_size, source_seq_length]
        # target.size() = [batch_size, target_seq_length]

        batch_size, max_len = target.size()
        target_vocab_size = self.decoder.tokens_num

        outputs = torch.zeros(max_len, batch_size, target_vocab_size, device=self.device)

        x_encoder, hidden_encoder, cell_encoder = self.encoder(source)

        hidden_encoder, cell_encoder = (hidden_encoder_conversion(hidden_encoder, device=self.device),
                                        hidden_encoder_conversion(cell_encoder, device=self.device))

        hidden = torch.zeros(hidden_encoder.size(0), hidden_encoder.size(1), self.decoder.hidden_size // 2,
                             device=self.device)
        cell = torch.zeros(hidden_encoder.size(0), hidden_encoder.size(1), self.decoder.hidden_size // 2,
                           device=self.device)

        for idx in range(len(hidden_encoder)):
            hidden[idx] = self.dropout_hidden_state(self.hidden_state_mappings[idx](hidden_encoder[idx]))
            cell[idx] = self.dropout_cell_state(self.cell_state_mappings[idx](cell_encoder[idx]))

        cur_hidden, cur_cell = torch.cat([hidden, hidden], dim=2), torch.cat([cell, cell], dim=2)

        cur_input = target[:, 0]

        for idx in range(1, max_len):
            output, cur_hidden, cur_cell = self.decoder(cur_input, cur_hidden, cur_cell, x_encoder, self.device)

            cur_hidden[:, :, self.decoder.hidden_size // 2:] += hidden
            cur_cell[:, :, self.decoder.hidden_size // 2:] += cell
            #  output.size() = [batch_size, output_dim]

            outputs[idx] = output

            teacher_force_flag = np.random.rand() < teacher_forcing_ration
            top1 = output.max(dim=1)[1]

            cur_input = (target[:, idx] if teacher_force_flag else top1)

        return outputs
