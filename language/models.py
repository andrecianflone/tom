import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import data
from allennlp.modules.elmo import Elmo, batch_to_ids

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ElmoEmbedding(nn.Module):
    """
    Custom Embedding class to handle funky ELMo

    ELMo does not have an embedding matrix, embeddings are generated on the
    fly, aka contextual embeddings. Under the hood, ELMo parses raw text and
    returns token embedding of size 1024, which is the average of 3 layers
    """
    def __init__(self, itos):
        super(ElmoEmbedding, self).__init__()

        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        print("Loading ELMo class, may take awhile if 1st time")
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
        self.itos = itos

    def forward(self, x):
        sentences = []
        for sent in x:
            sentences.append([self.itos[t] for t in sent])
        character_ids = batch_to_ids(sentences).to(device)
        embeddings = self.elmo(character_ids)
        emb = embeddings['elmo_representations'][0]
        return emb

class RNNModel(nn.Module):
    """
    Simple model from pytorch repo
    Container module with an encoder, a recurrent module, and a decoder.
    """
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                                                            tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers,
                                                            dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh',
                                                'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was
                                supplied, options are ['LSTM', 'GRU',
                                'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity,
                                                            dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in: "Using the Output Embedding to Improve
        # Language Models" (Press & Wolf 2016) https://arxiv.org/abs/1608.05859
        # and "Tying Word Vectors and Word Classifiers: A Loss Framework for
        # Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError("""When using the tied flag, nhid must be
                                                            equal to emsize""")
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1),
                                                            output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)),hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


# A lot of the Seq2Seq code based on:
# https://github.com/bentrevett/pytorch-seq2seq
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout,
                                            str_to_idx, itos):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.dropout = nn.Dropout(dropout)
        self.itos = itos
        self.str_to_idx = str_to_idx

    def forward(self, src, src_len):
        """
        Args:
            src: [src sent len, batch size]
            src_len: [src sent len]
        """
        # Swap integers for embeddings -> [sent len, batch size, emb dim]
        embedded = self.dropout(self.embedding(src))

        # Pack and run through RNN
        # Outputs -> [sent len, batch size, encoder dim * 2]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, encoder_outputs, mask):
        """
        Args:
            hidden: [batch size, dec hid dim]
            encoder_outputs: [src sent len, batch size, enc hid dim * 2]
            mask: [batch size, src sent len]
        """
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #hidden = [batch size, src sent len, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        #energy = [batch size, src sent len, dec hid dim]
        energy = energy.permute(0, 2, 1)
        #energy = [batch size, dec hid dim, src sent len]
        #v = [dec hid dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        #v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)
        #attention = [batch size, src sent len]
        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout,
                                        attention, str_to_idx, itos):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim,
                                                                    output_dim)
        self.dropout = nn.Dropout(dropout)
        self.itos = itos
        self.str_to_idx = str_to_idx

    def forward(self, input, hidden, encoder_outputs, mask):
        """
        Args:
            input: [batch size]
            hidden: [batch size, dec hid dim]
            encoder_outputs: [src sent len, batch size, enc hid dim * 2]
            mask: [batch size, src sent len]
        """

        input = input.unsqueeze(0)
        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]
        a = self.attention(hidden, encoder_outputs, mask)
        #a = [batch size, src sent len]
        a = a.unsqueeze(1)
        #a = [batch size, 1, src sent len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch size, enc hid dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output = [sent len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]

        #sent len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        output = self.out(torch.cat((output, weighted, embedded), dim = 1))
        #output = [bsz, output dim]
        return output, hidden.squeeze(0), a.squeeze(1)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

class Seq2Seq(nn.Module):
    def __init__(self, args, encoder, decoder, pad_idx, sos_idx, eos_idx, device,
                    use_embeddings, loaded_vectors, trainable_embeddings):
        """
        Args:
            loaded_vectors (dict) : all words in data mapped to embedding
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device
        enc_hid_dim = encoder.enc_hid_dim
        dec_hid_dim = decoder.dec_hid_dim
        # FC layer to project encoder hidden to decoder hidden
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        # Initialize weights
        self.apply(init_weights)

        # If use embeddings, prepare enc/dec embeddings
        if use_embeddings and args.embedding_type == "elmo":
            self.encoder.embedding = ElmoEmbedding(self.encoder.itos)
            self.decoder.embedding = ElmoEmbedding(self.decoder.itos)

        elif use_embeddings:
            # np array of idx-to-embedding
            enc_emb = data.vectors_lookup(loaded_vectors,encoder.str_to_idx,
                                                            encoder.emb_dim)
            self.encoder.embedding.weight.data.copy_(torch.from_numpy(enc_emb))

            dec_emb = data.vectors_lookup(loaded_vectors,decoder.str_to_idx,
                                                            decoder.emb_dim)
            self.decoder.embedding.weight.data.copy_(torch.from_numpy(dec_emb))

        # If not trainable, no grad embeddings
        if not trainable_embeddings:
            self.encoder.embedding.requires_grad = False
            self.decoder.embedding.requires_grad = False

    def create_mask(self, src):
        mask = (src != self.pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        """
        Args:
            src = [src sent len, batch size]
            src_len = [batch size]
            trg = [trg sent len, batch size]
            teacher_forcing_ratio is probability to use teacher forcing e.g. if
            teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the
            time
        """
        if trg is None:
            assert teacher_forcing_ratio == 0, "Must be zero during inference"
            inference = True
            trg = torch.zeros((100, src.shape[1])).long().fill_(self.sos_idx).to(src.device)
        else:
            inference = False

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        #tensor to store attention
        attentions = torch.zeros(max_len, batch_size, src.shape[0]).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # Project Encoder output hidden to Decoder hidden
        hidden = self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        hidden = torch.tanh(hidden)

        #first input to the decoder is the <sos> tokens
        output = trg[0,:]

        mask = self.create_mask(src)
        #mask = [batch size, src sent len]

        for t in range(1, max_len):
            output, hidden, attention = self.decoder(output, hidden, encoder_outputs, mask)
            outputs[t] = output
            attentions[t] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
            if inference and output.item() == self.eos_idx:
                return outputs[:t], attentions[:t]

        return outputs, attentions

