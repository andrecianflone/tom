"""
Seq2Seq with Attention, packed seq and masking, based on:
https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%20and%20Inference.ipynb

See all models, including transformer here:
https://github.com/bentrevett/pytorch-seq2seq
"""
import random
import math
import time
import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import data
from data import tokenize_en
from models import Encoder, Decoder, Attention, Seq2Seq
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, src_len = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output, attention = model(src, src_len, trg)
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len = batch.src
            trg = batch.trg

            output, attention = model(src, src_len, trg, 0) #turn off teacher forcing
            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def get_data_model(args):
    # Get data
    train_iterator, valid_iterator, test_iterator, src, trg = data.load_naive(args)
    # print(f"Number of training examples: {len(train_data.examples)}")
    # print(f"Number of validation examples: {len(valid_data.examples)}")
    # print(f"Number of testing examples: {len(test_data.examples)}")

    # Create model
    input_dim = len(src.vocab)
    output_dim = len(trg.vocab)
    pad_idx = src.vocab.stoi['<pad>']
    sos_idx = trg.vocab.stoi['<sos>']
    eos_idx = trg.vocab.stoi['<eos>']
    attn = Attention(args.enc_dim, args.dec_dim)
    enc = Encoder(input_dim, args.emb_dim, args.enc_dim, args.dec_dim,
                                                                args.dropout)
    dec = Decoder(output_dim, args.emb_dim, args.enc_dim, args.dec_dim,
                                                            args.dropout, attn)
    model = Seq2Seq(enc, dec, pad_idx, sos_idx, eos_idx, device).to(device)

    print(f'The model has {utils.count_parameters(model):,} trainable parameters')
    return train_iterator, valid_iterator, test_iterator, model, src, trg

def inference(args):
    """
    Load data,  model, and generate a sentence based on context.
    """
    train_iterator, valid_iterator, test_iterator, model, src, trg =\
                                                        get_data_model(args)
    model.load_state_dict(torch.load(args.save_path))

    # Sample sentence from the test set
    sent = "A polite thief was making robberies in the small town. People would wake up and find things missing and their house clean. One one occasion the polite thief left a note of apology."

    # Generate prediction
    gen_sent, attn = generate_sentence(model, sent, src, trg)
    print(f"Original context: \n {sent} \n")
    print(f"Generated: \n {gen_sent} \n")

def generate_sentence(model, sentence, src, trg):
    """
    Given a string, generate a follow up sentence
    Args:
        sentence: (string)
    """
    model.eval()
    tokenized = tokenize_en(sentence)
    tokenized = ['<sos>'] + [t.lower() for t in tokenized] + ['<eos>']
    numericalized = [src.vocab.stoi[t] for t in tokenized]
    sentence_length = torch.LongTensor([len(numericalized)]).to(device)
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    translation_tensor_logits, attention = model(tensor, sentence_length, None, 0)
    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
    translation = [trg.vocab.itos[t] for t in translation_tensor]
    translation, attention = translation[1:], attention[1:]
    return translation, attention

def main(args):
    #####################################
    # DATA and MODEL
    #####################################
    train_iterator, valid_iterator, test_iterator, model, src, trg =\
                                                        get_data_model(args)

    #####################################
    # TRAIN LOOP
    #####################################
    best_valid_loss = float('inf')
    optimizer = optim.Adam(model.parameters())
    pad_idx = src.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    for epoch in range(args.max_epochs):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, args.grad_clip)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            if not os.path.exists(folder):
                os.makedirs(folder)
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.save_path)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    #####################################
    # POST-TRAINING EVAL
    #####################################
    model.load_state_dict(torch.load(args.save_path))
    test_loss = evaluate(model, test_iterator, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--enc_dim', type=int, default=256, metavar='N',
                        help='encoder hidden size')
    parser.add_argument('--dec_dim', type=int, default=256, metavar='N',
                        help='decoder hidden size')
    parser.add_argument('--emb_dim', type=int, default=200, metavar='N',
                        help='embedding size')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient norm clip')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Whether to save the model while training')
    parser.add_argument('--saved_model_name', type=str, default='naive.pt')
    parser.add_argument('--max_epochs', type=int, default=15,
                        help='upper epoch limit')
    parser.add_argument('--prepared_data', type=str,
                        default='.data/64_naive_data.pickle',
                        help='path of prepared data')
    parser.add_argument('--label_cond', action='store_true', default=True,
                        help='Label conditioning')

    parser.add_argument('--generate', action='store_true', default=False,
                        help='Inference test')
    # Datasets
    parser.add_argument('--with_emotions', action='store_true', default=False,
                        help='Use the source datasets with emotions')

    args = parser.parse_args()

    # Model save path
    folder = "saved_models"
    args.save_path = os.path.join(folder, args.saved_model_name)

    # Maybe source with emotions
    if args.with_emotions:
        print("Will train with emotions")
        args.src_ext = ".src_meta"
    else:
        print("Will not train with emotions")
        args.src_ext = ".src"
    args.trg_ext = ".trg"

    args.device = device
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Start
    if args.generate:
        inference(args)
    else:
        main(args)
