"""
LM on the Naive Psych story dataset

Beam search:
    AllenNLP: https://allenai.github.io/allennlp-docs/api/allennlp.nn.beam_search.html
    OpenNMT: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
"""
import math
import time
import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import data
from data import tokenize_en
import utils
import numpy as np
from tqdm import trange

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args, model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        # For debugging, check if max num of batches
        if args.max_batches is not None and args.max_batches > i:
            break
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

def evaluate_ppl_gpt(args):
    """
    Evaluate on raw text, use this with GPT which has its own tokenizer
    """
    if args.expanded_dataset:
        path = ".data/stories/story_commonsense/torchtext_expanded"
    else:
        path = ".data/stories/story_commonsense/torchtext"
    # Data
    test_src = [line.rstrip('\n') for line in open(path+"/test.src")]
    test_trg = [line.rstrip('\n') for line in open(path+"/test.trg")]

    # Model
    enc = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    loss = 0
    batch_size = 1

    for i in trange(len(test_src)):
        src, trg = test_src[i], test_trg[i]
        context = enc.encode(src)
        target = enc.encode(trg)
        length = len(target)

        # Generate prediction
        out = utils.sample_sequence(model, length, batch_size=1,
                                    context=context, top_k=10, device=device)
        out = out[:, len(context):]

        # Get model loss
        target = torch.tensor([target]).to(device)
        with torch.no_grad():
            #pred, past  = model(out)
            l = model(out, lm_labels=target)
            loss += float(l)
    av_loss = loss / len(loss)
    print(f"ppl: {math.exp(av_loss):.04f}")

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

def inference(args):
    """
    Load data,  model, and generate a sentence based on context.
    """
    train_iterator, valid_iterator, test_iterator, src, trg, vec=\
                                                        data.get_data(args)
    model = utils.create_seq2seq_model(args, src, trg, vec)
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

@utils.log_time_delta
def main(args):
    # Get data and model
    train_iterator, valid_iterator, test_iterator, src, trg, vec =\
                                                        data.get_data(args)
    model = utils.create_seq2seq_model(args, src, trg, vec)

    best_valid_loss = float('inf')
    best_valid_epoch = 0
    optimizer = optim.Adam(model.parameters())
    pad_idx = src.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    # Main loop
    for epoch in range(args.num_epochs):
        start_time = time.time()

        train_loss = train(args, model, train_iterator, optimizer,
                                                    criterion, args.grad_clip)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            if not os.path.exists(folder):
                os.makedirs(folder)
            best_valid_loss = valid_loss
            best_valid_epoch = epoch
            torch.save(model.state_dict(), args.save_path)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    # Post training eval on test
    model.load_state_dict(torch.load(args.save_path))
    test_loss = evaluate(model, test_iterator, criterion)
    print('****RESULTS****')
    print(f'| Best Val. Loss: {best_valid_loss:.3f} | Best Val. PPL: {math.exp(best_valid_loss):7.3f} | At epoch: {best_valid_epoch} ')
    print(f'| Test Loss with best val model: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | At epoch: {best_valid_epoch} ')

if __name__ == '__main__':
    print(f"Script launched with :\n{' '.join(sys.argv)}")
    parser = argparse.ArgumentParser(description='')
    add = parser.add_argument

    # Training settings
    add('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    add('--seed', type=int, default=1111,
                        help='random seed')
    add('--enc_dim', type=int, default=256, metavar='N',
                        help='encoder hidden size')
    add('--dec_dim', type=int, default=256, metavar='N',
                        help='decoder hidden size')
    add('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    add('--grad_clip', type=float, default=1.0,
                        help='Gradient norm clip')
    add('--num_epochs', type=int, default=15,
                        help='upper epoch limit')
    add('--max_batches', type=int, default=None,
                        help='Max batches per epoch, for debugging (default None)')

    # model
    add('--save', action='store_true', default=True,
                        help='Whether to save the model while training')
    add('--saved_model_name', type=str, default='naive.pt')


    # Task
    add('--task', default='lm_train', choices=['lm_train', 'lm_test',
                        'generate', 'classification'],
                        help='Which task to do (default: %(default)s)')
    add('--emb_dim', type=int, default=300, metavar='N',
                        help='embedding size')
    add('--label_cond', action='store_true', default=True,
                        help='Label conditioning')
    add('--generate', action='store_true', default=False,
                        help='Inference test')
    # Embeddings
    add('--embedding_type', default='None', choices=['glove', 'elmo', 'gpt', 'bert'],
                        help='Embedding type (default: %(default)s)')
    add('--use_pretrained_embeddings', action='store_true',
                default=False, help='Use pretrained embeddings such as Glove')
    add('--trainable_embeddings', action='store_true',
                    default=False, help='Should embeddings should trainable')
    add('--embeddings_path', type=str,
                        default='.data/embeddings/glove.6B.300d.txt',
                        help='Glove embeddings path')

    # Datasets
    add('--with_emotions', action='store_true', default=False,
                        help='Use the source datasets with emotions')
    add('--single_vocab', action='store_true', default=True,
                        help='Same vocab for encoder and decoder')
    add('--prepared_data', type=str, default='.data/naive_data.pickle',
                        help='path of prepared data')
    add('--expanded_dataset', action='store_true', default=False,
                        help='Expanded Naive dataset')

    args = parser.parse_args()

    # Model save path
    folder = "saved_models"
    args.save_path = os.path.join(folder, args.saved_model_name)

    # Maybe source with emotions
    if args.with_emotions:
        print("Not using emotions")
        args.src_ext = ".src_meta"
    else:
        print("Using emotions")
        args.src_ext = ".src"
    args.trg_ext = ".trg"

    args.device = device
    # Set seeds
    # random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Start
    if args.task == "generate":
        inference(args)
    elif args.task == "lm_train":
        main(args)
    elif args.task == "lm_test":
        print("Evaluating test set with GPT2")
        evaluate_ppl_gpt(args)
    elif args.task == "classificaiton":
        classify(args)

