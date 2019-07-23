"""
LM on the Naive Psych story dataset

Beam search:
    AllenNLP:
      https://allenai.github.io/allennlp-docs/api/allennlp.nn.beam_search.html
    OpenNMT:
      https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam.py
"""
import math
import time
import os
import sys
import argparse
import pickle
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import data
from data import tokenize_en
import utils
import models
import numpy as np
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_cl(args, model, maslow_it, reiss_it, optimizer, criterion, clip):
    """ Fine tune model for classification """
    model.train()
    epoch_loss = 0
    ma_pred = []
    ma_true = []
    re_pred = []
    re_true = []
    # reiss_iterator = iter(reiss_it)

    #TODO: add tqdm here, show progress bar

    # Note if iterators not same length, one it will end early
    for i, (batch_ma, batch_re) in enumerate(zip(maslow_it, reiss_it)):
        # For debugging, check if max num of batches
        if args.max_batches is not None and i > args.max_batches:
            break

        # Maslow
        text, text_len = batch_ma.text
        label = batch_ma.label
        optimizer.zero_grad()
        output = model(text, text_len, label, task='maslow')
        # sys.exit(1)
        loss1 = criterion(output, label)
        _, predicted = torch.max(output.data, 1)
        ma_pred.extend(predicted.tolist())
        ma_true.extend(label.tolist())

        # Reiss
        # batch = next(reiss_iterator)
        text, text_len = batch_re.text
        label = batch_re.label
        optimizer.zero_grad()
        output = model(text, text_len, label, task='reiss')
        loss2 = criterion(output, label)
        _, predicted = torch.max(output.data, 1)
        re_pred.extend(predicted.tolist())
        re_true.extend(label.tolist())

        loss = loss1 + loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        ## Accuracy
        # _, predicted = torch.max(output.data, 1)
        # correct += (predicted==label).sum().item()
        # total += label.size(0)

    total_loss = epoch_loss / len(maslow_it)
    return  total_loss, ma_pred, ma_true, re_pred, re_true

def train_lm(args, model, iterator, optimizer, criterion, clip):
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

    print("Evaluating test set with GPT2")
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
            l = model(out, labels=target)
            loss += float(l)
    av_loss = loss / len(loss)
    print(f"ppl: {math.exp(av_loss):.04f}")

def evaluate_cl(model, maslow_it, reiss_it):
    """
    Evaluate the model for a classification task
    Return loss and accuracy
    """
    model.eval()
    epoch_loss = 0
    ma_pred = []
    ma_true = []
    re_pred = []
    re_true = []

    with torch.no_grad():
        for i, batch in enumerate(maslow_it):
            text, text_len = batch.text
            label = batch.label
            output = model(text, text_len, label, task='maslow')
            loss = criterion(output, label)

            _, predicted = torch.max(output.data, 1)
            ma_pred.extend(predicted.tolist())
            ma_true.extend(label.tolist())
            epoch_loss += loss.item()

        for i, batch in enumerate(reiss_it):
            text, text_len = batch.text
            label = batch.label
            output = model(text, text_len, label, task='reiss')

            _, predicted = torch.max(output.data, 1)
            re_pred.extend(predicted.tolist())
            re_true.extend(label.tolist())
            epoch_loss += loss.item()

    total_loss = epoch_loss / len(maslow_it)
    return  total_loss, ma_pred, ma_true, re_pred, re_true

def evaluate_lm(model, iterator, criterion):
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
                                                        data.get_lm_data(args)
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
    translation_tensor_logits,attention = model(tensor,sentence_length,None,0)
    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
    translation = [trg.vocab.itos[t] for t in translation_tensor]
    translation, attention = translation[1:], attention[1:]
    return translation, attention

def most_probable_label(model, label, label_query, dictionary, past, tokenizer):

    # Encode labels
    lowest_loss = float('inf')
    pred_label = -1
    true_label = dictionary.word2idx[label]

    for i, lbl in enumerate(dictionary.idx2word):
        current_label = i
        # Encode for GPT
        lbl = label_query + " " + lbl
        text = torch.tensor([tokenizer.encode(lbl)])
        outputs = model(text, labels=text, past=past)
        loss = outputs[:1]
        loss = float(loss[0])
        if loss < lowest_loss:
            lowest_loss = loss
            pred_label = i

    # Return arg max and true label
    return pred_label, true_label

def evaluate_zero_shot(args, model, tokenizer, path, src_query, trg_query):
    """
    Evaluate the model for a zero-shot classification task
    Return loss and accuracy
    """
    model.eval()
    pred_ls = []
    true_ls = []

    #### Data
    test_src = [line.rstrip('\n') for line in open(path+"/test.src")]
    test_trg = [line.rstrip('\n') for line in open(path+"/test.trg")]

    # Shuffle in case of short eval
    src_shuf = []
    trg_shuf = []
    index_shuf = list(range(len(test_src)))
    shuffle(index_shuf)
    for i in index_shuf:
        src_shuf.append(test_src[i])
        trg_shuf.append(test_trg[i])
    test_src = src_shuf
    test_trg = trg_shuf

    # Targets dictionary
    dictionary = data.Dictionary()
    for l in test_trg:
        dictionary.add_word(l)

    n_samples = len(test_src)
    if args.max_batches is not None and args.max_batches < n_samples:
        n_samples = args.max_batches

    # for i in trange(len(test_src)):
    for i in trange(n_samples):
        src, trg = test_src[i], test_trg[i]
        src += src_query
        # Get context hidden states once to speed up eval
        context = torch.tensor([tokenizer.encode(src)])
        pred, past = model(context)
        mp, true_lbl = most_probable_label(model, trg, trg_query, dictionary,
                                                            past, tokenizer)
        pred_ls.append(mp)
        true_ls.append(true_lbl)

    return  pred_ls, true_ls

def zero_shot_gpt2(args):
    print('Get model')
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print("Evaluating Maslow test set with GPT2")
    path = ".data/stories/story_commonsense/torchtext_class/maslow/"
    src_query = " That made them"
    trg_query = " feel" # need to split due to offset in loss
    ma_t_pred, ma_t_true = \
        evaluate_zero_shot(args, model, tokenizer, path, src_query, trg_query)

    # Maslow results
    t_acc = accuracy_score(ma_t_true, ma_t_pred)
    t_f1 = f1_score(ma_t_true, ma_t_pred, average='macro')
    t_p = precision_score(ma_t_true, ma_t_pred, average='macro')
    t_r = recall_score(ma_t_true, ma_t_pred, average='macro')
    print('Maslow')
    print(f'\t Test | acc: {t_acc:7.4f} | f1: {t_f1:7.4f} | prec: {t_p:7.4f} | rec: {t_r:7.4f}')

    print("Evaluating Reiss test set with GPT2")
    path = ".data/stories/story_commonsense/torchtext_class/reiss/"
    src_query = " They did this to"
    trg_query = " to" # need to split due to offset in loss
    re_t_true, re_t_pred = \
        evaluate_zero_shot(args, model, tokenizer, path, src_query, trg_query)

    # Reiss results
    t_acc = accuracy_score(re_t_true, re_t_pred)
    t_f1 = f1_score(re_t_true, re_t_pred, average='macro')
    t_p = precision_score(re_t_true, re_t_pred, average='macro')
    t_r = recall_score(re_t_true, re_t_pred, average='macro')
    print('Reiss')
    print(f'\t Test | acc: {t_acc:7.4f} | f1: {t_f1:7.4f} | prec: {t_p:7.4f} | rec: {t_r:7.4f}')

def main_classification(args):
    print('Get data and model')
    ma_iterators, reiss_iterators, text, vec = data.get_cl_data(args)
    maslow_train_it, maslow_valid_it, maslow_test_it, maslow_label=ma_iterators
    reiss_train_it, reiss_valid_it, reiss_test_it, reiss_label= reiss_iterators

    # Number of labels per task:
    classes = [len(maslow_label.vocab), len(reiss_label.vocab)]
    if args.model == 'seq2seq':
        model = utils.create_seq2seq_model_cl(args, text, text, vec,
                                                    maslow_label, reiss_label)
    elif args.model == 'gpt2':
        model = models.GPT2Classifier(classes, args.gpttokenizer).to(device)

    best_valid_loss = float('inf')
    best_valid_epoch = 0
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    print('Training starting...')
    # Main loop
    for epoch in range(args.num_epochs):
        start_time = time.time()

        # Training
        train_loss, ma_tr_pred, ma_tr_true, re_tr_pred, re_tr_true = \
                train_cl(args, model, maslow_train_it, reiss_train_it,
                        optimizer,criterion, args.grad_clip)

        # Validation
        valid_loss, ma_v_pred, ma_v_true, re_v_pred, re_v_true = \
                evaluate_cl(model, maslow_valid_it, reiss_valid_it, criterion)
        # Test
        test_loss, ma_t_pred, ma_t_true, re_t_pred, re_t_true = \
                evaluate_cl(model, maslow_test_it, reiss_test_it, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_epoch = epoch

        # Maslow
        tr_acc = accuracy_score(ma_tr_true, ma_tr_pred)
        v_acc = accuracy_score(ma_v_true, ma_v_pred)
        v_f1 = f1_score(ma_v_true, ma_v_pred, average='macro')
        v_p = precision_score(ma_v_true, ma_v_pred, average='macro')
        v_r = recall_score(ma_v_true, ma_v_pred, average='macro')
        t_acc = accuracy_score(ma_t_true, ma_t_pred)
        t_f1 = f1_score(ma_t_true, ma_t_pred, average='macro')
        t_p = precision_score(ma_t_true, ma_t_pred, average='macro')
        t_r = recall_score(ma_t_true, ma_t_pred, average='macro')

        print('Maslow')
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | acc: {tr_acc:7.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f} | acc: {v_acc:7.4f} | f1: {v_f1:7.4f} | prec: {v_p:7.4f} | rec: {v_r:7.4f}')
        print(f'\t Test Loss: {test_loss:.4f} | acc: {t_acc:7.4f} | f1: {t_f1:7.4f} | prec: {t_p:7.4f} | rec: {t_r:7.4f}')

        # Reiss
        tr_acc = accuracy_score(re_tr_true, re_tr_pred)
        v_acc = accuracy_score(re_v_true, re_v_pred)
        v_f1 = f1_score(re_v_true, re_v_pred, average='macro')
        v_p = precision_score(re_v_true, re_v_pred, average='macro')
        v_r = recall_score(re_v_true, re_v_pred, average='macro')
        t_acc = accuracy_score(re_t_true, re_t_pred)
        t_f1 = f1_score(re_t_true, re_t_pred, average='macro')
        t_p = precision_score(re_t_true, re_t_pred, average='macro')
        t_r = recall_score(re_t_true, re_t_pred, average='macro')

        print('Reiss')
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | acc: {tr_acc:7.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f} | acc: {v_acc:7.4f} | f1: {v_f1:7.4f} | prec: {v_p:7.4f} | rec: {v_r:7.4f}')
        print(f'\t Test Loss: {test_loss:.4f} | acc: {t_acc:7.4f} | f1: {t_f1:7.4f} | prec: {t_p:7.4f} | rec: {t_r:7.4f}')

@utils.log_time_delta
def main_lm(args):
    # Get data and model
    train_iterator, valid_iterator, test_iterator, src, trg, vec =\
                                                        data.get_lm_data(args)
    model = utils.create_seq2seq_model(args, src, trg, vec)

    best_valid_loss = float('inf')
    best_valid_epoch = 0
    optimizer = optim.Adam(model.parameters())
    pad_idx = src.vocab.stoi['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    # Main loop
    for epoch in range(args.num_epochs):
        start_time = time.time()

        train_loss = train_lm(args, model, train_iterator, optimizer,
                                                    criterion, args.grad_clip)
        valid_loss = evaluate_lm(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            if not os.path.exists(folder):
                os.makedirs(folder)
            best_valid_loss = valid_loss
            best_valid_epoch = epoch
            torch.save(model.state_dict(), args.save_path)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):7.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f} |  Val. PPL: {math.exp(valid_loss):7.4f}')

    # Post training eval on test
    model.load_state_dict(torch.load(args.save_path))
    test_loss = evaluate(model, test_iterator, criterion)
    print('****RESULTS****')
    print(f'| Best Val. Loss: {best_valid_loss:.4f} | Best Val. PPL: {math.exp(best_valid_loss):7.4f} | At epoch: {best_valid_epoch} ')
    print(f'| Test Loss with best val model: {test_loss:.4f} | Test PPL: {math.exp(test_loss):7.4f} | At epoch: {best_valid_epoch} ')

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
                        help='Max batches per epoch, to debug (default None)')

    # model
    add('--model', default='seq2seq', choices=['seq2seq', 'gpt2'],
                        help='Type of model (default: %(default)s)')
    add('--save', action='store_true', default=True,
                        help='Whether to save the model while training')
    add('--saved_model_name', type=str, default='naive.pt')

    # Task
    add('--task', default='lm_train', choices=['lm_train', 'lm_test',
                        'generate', 'classification', 'zero_shot'],
                        help='Which task to do (default: %(default)s)')
    add('--emb_dim', type=int, default=300, metavar='N',
                        help='embedding size')
    add('--label_cond', action='store_true', default=True,
                        help='Label conditioning')
    add('--generate', action='store_true', default=False,
                        help='Inference test')
    # Embeddings
    add('--embedding_type', default='None',
                        choices=['glove', 'elmo', 'gpt', 'bert'],
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

    add('--tokenizer', default='spacy', choices=['raw', 'spacy', 'gpt2'],
                        help='Tokenizer for iterators (default: %(default)s)')

    args = parser.parse_args()

    # Model save path
    folder = "saved_models"
    args.save_path = os.path.join(folder, args.saved_model_name)

    # GPT2 special settings
    if args.model == 'gpt2':
        args.tokenizer = 'gpt2'
        args.gpt_maslowfield, args.gpt_reissfield, args.gpttokenizer = \
                               models.GPT2Classifier.field(args.task)

    # Maybe source with emotions
    if args.with_emotions:
        print("Using emotions")
        args.src_ext = ".src_meta"
    else:
        print("Not using emotions")
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
        main_lm(args)
    elif args.task == "lm_test":
        evaluate_ppl_gpt(args)
    elif args.task == "classification":
        main_classification(args)
    elif args.task == "zero_shot":
        zero_shot_gpt2(args)
