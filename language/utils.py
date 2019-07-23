import torch
import torch.nn.functional as F
from functools import wraps
import time
from datetime import datetime
from models import Encoder, Decoder, Attention, Seq2Seq
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

def get_batch(args, source, i):
    """
    get_batch subdivides the source data into chunks of length args.bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    """
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def batchify(data, bsz):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more
    efficient batch processing.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def log_time_delta(function):
    """
    Decorate `function` to compute & print execute time
    """
    @wraps(function)
    def _deco(*args, **kwargs):
        start = datetime.now()
        ret = function(*args, **kwargs)
        end = datetime.now()
        s = (end - start).total_seconds()
        hours, remainder = divmod(s, 3600)
        minutes, seconds = divmod(remainder, 60)
        print('{} ran in time: {:02}:{:02}:{:02}'.format(function.__name__,
                                        int(hours), int(minutes), int(seconds)))
        return ret
    return _deco

def log_seconds_delta(func):
    """
    Decorate `func` to compute & print execute time
    """
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco

def create_seq2seq_model(args, src, trg, loaded_vectors):
    """

    Args:
        src: Field
        trg: Field
    """
    input_dim = len(src.vocab)
    output_dim = len(trg.vocab)
    pad_idx = src.vocab.stoi['<pad>']
    sos_idx = trg.vocab.stoi['<sos>']
    eos_idx = trg.vocab.stoi['<eos>']
    attn = Attention(args.enc_dim, args.dec_dim)
    enc = Encoder(input_dim, args.emb_dim, args.enc_dim, args.dec_dim,
                                args.dropout, src.vocab.stoi, src.vocab.itos)
    dec = Decoder(output_dim, args.emb_dim, args.enc_dim, args.dec_dim,
                            args.dropout, attn, trg.vocab.stoi, trg.vocab.itos)
    model = Seq2Seq(args, enc, dec, pad_idx, sos_idx, eos_idx, device,
                                args.use_pretrained_embeddings, loaded_vectors,
                                args.trainable_embeddings).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')

def create_seq2seq_model_cl(args, src, trg, loaded_vectors, maslow_label,
                                                                reiss_label):
    """
    Seq2Seq for classification
    """
    model = create_seq2seq_model(args, src, trg, loaded_vectors)
    model.load_state_dict(torch.load(args.save_path))
    model.mode_classification(\
            maslow_classes=len(maslow_label.vocab.itos),
            reiss_classes=len(reiss_label.vocab.itos))
    return model

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in range(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output
