# tom

## Installation

Simply run

```
bash setup.sh
```

If `allennlp` fails with error that it cannot uninstall a package, simply manually remove that package from the `site-packages` directory.


## Useful commands
```
cd language/

# Create the dataset for language modeling
python data.py --create_naive

# Create the expanded version for language modeling
python data.py --create_naive_expanded

# Create the classification dataset
python data.py --create_classification_files

```
## Dataset

### Fifth sentence stats
Number of stories where fifth sentence contains an example of

|                  |   train |   val |   test |
|:-----------------|--------:|------:|-------:|
| emotion_plutchik |       0 |  2220 |   1999 |
| emotion_text     |    9129 |  2223 |   2002 |
| motiv_maslow     |       0 |  2114 |   1958 |
| motiv_reiss      |       0 |  2086 |   1916 |
| motiv_text       |    6330 |  2114 |   1959 |
| story_count      |    9885 |  2483 |   2370 |

### Classification dataset stats
*Maslow*
|                  |   train |   val |   test |
|:-----------------|--------:|------:|-------:|
| esteem           |    1605 |   388 |   1626 |
| love             |    1671 |   393 |   1745 |
| physiological    |    1028 |   264 |   1286 |
| spiritual growth |    1402 |   362 |   1762 |
| stability        |    2162 |   559 |   2683 |
| tot              |    7868 |  1966 |   9102 |

*Reiss*
|             |   train |   val |   test |
|:------------|--------:|------:|-------:|
| approval    |     364 |    85 |    400 |
| belonging   |      30 |     7 |     24 |
| competition |     489 |   119 |    520 |
| contact     |     663 |   164 |    653 |
| curiosity   |     652 |   164 |    798 |
| family      |     538 |   129 |    585 |
| food        |     611 |   136 |    858 |
| health      |     430 |   113 |    524 |
| honor       |     154 |    35 |    189 |
| idealism    |     156 |    37 |    208 |
| indep       |     444 |   108 |    460 |
| order       |     661 |   186 |    870 |
| power       |     233 |    54 |    214 |
| rest        |     235 |    51 |    223 |
| romance     |     290 |    69 |    315 |
| savings     |     591 |   153 |    692 |
| serenity    |     218 |    61 |    339 |
| status      |     458 |   125 |    422 |
| tranquility |     502 |   133 |    623 |
| tot         |    7719 |  1929 |   8917 |

## Useful References
ELMo:
- [intro](https://allennlp.org/elmo), [paper](https://www.aclweb.org/anthology/N18-1202), [quick tutorial](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md), [long tutorial](http://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/)


