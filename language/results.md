
### Effect of embedding size
```
python new_main.py --emb_dim 200
```

emb size | epoch | val ppl | param count
---      | ---   | ---     | ---
 50      | 7     | 180.818 | 5,348,255
100      | 7     | 181.707 |
200      | 6     | 180     |
300      | 6     | 182.215 | 10,086,755
400      | 6     | 182.223 | 11,982,155

### Effect of embedding size with emotions
```
python new_main.py --with_emotions --emb_dim 50 --saved_model_name naive_emo.pt
```

emb size | epoch | val ppl | param count
---      | ---   | ---     | ---
 50      | 5     | 186.682 | 5,425,405
100      |       |         |
200      | 5     | 185.554 | 8,499,955
300      |       |         |
400      | 5     | 182.378 | 12,599,355
600      | 4     | 181.703 | 16,698,755

### Effect of Pretrained embeddings
Using Glove embeddings. By default, pretrained embeddings are fixed
```
python main.py --use_pretrained_embeddings --prepared_data .data/naive_data_emb_100.pickle --embeddings_path .data/embeddings/glove.6B.100d.txt --saved_model_name naive_100d.pt
```

emb size       | epoch | val ppl | test ppl | param count
---            | ---   | ---     | ---      | ---
50             | 6     | 164.726 | 167.316  | 5,348,255
100            | 6     | 166.119 | 168.330  | 7,295,955
200            | 3     | 165.831 | 168.423  | 8,191,355
300            | 3     | 157.842 | 159.932  | 10,086,755
300-840B Glove | 3     | 158.089 | 161.658  | 11,086,755
1024-ELMo      | 4     | 159.842 | 160.970  | 10,867,123

`301-840B` are the 300d embeddings trained on 840B tokens.

### Expanded
Number of training examples: 39540
Number of validation examples: 9932
Number of testing examples: 9480

- Expanded dataset
params:  17,479,109
| Best Val. Loss: 5.287 | Best Val. PPL: 197.695 | At epoch: 2
| Test Loss with best val model: 5.277 | Test PPL: 195.845 | At epoch: 2

Expanded, with emotions
params: 17,479,109
| Best Val. Loss: 5.293 | Best Val. PPL: 198.854 | At epoch: 2
| Test Loss with best val model: 5.291 | Test PPL: 198.584 | At epoch: 2

Expanded dataset with embeddings
| Best Val. Loss: 4.868 | Best Val. PPL: 130.030 | At epoch: 2
| Test Loss with best val model: 4.869 | Test PPL: 130.212 | At epoch: 2

Expanded dataset, with embeddings, with emotions
params: 10,549,655
| Best Val. Loss: 4.865 | Best Val. PPL: 129.685 | At epoch: 1
| Test Loss with best val model: 4.872 | Test PPL: 130.645 | At epoch: 1

 Expanded with ELMo
main.py --expanded_dataset --use_pretrained_embeddings --embedding_type elmo --prepared_data .data/naive_data_elmo.pickle --saved_model_name elmo_expanded.pt --emb_dim 1024Will not train with emotions

Epoch: 04 | Time: 46m 39s
Train Loss: 4.020 | Train PPL:  55.700
Val. Loss: 4.822 |  Val. PPL: 124.185


GPT2 av ppl on normal dataset:
```
python  main.py --task lm_test
```
10.2006.

GPT2 av ppl on expanded  dataset:
```
python  main.py --task lm_test --expanded_dataset
```
10.2123

### Classification
Validation set f1 is in the 60s, yet test is quite low. Should probably
increase the validation set size.

| Model                            | Maslow F1 | Reiss F1 |
| :---                             | ---:      | ---:     |
| Pretrained LM Glove 300          | 51.67     | 25.89    |
| Pretrained Expanded LM Glove 300 | 53.32     | 27.08    |
| Pretrained LM ELMO               | 49.22     | 25.46    |

