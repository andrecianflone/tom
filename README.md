# tom

## Installation

Simply run

```
bash setup.sh
```

If `allennlp` fails with error that it cannot uninstall a package, simply manually remove that package from the `site-packages` directory.

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

## Useful References

ELMo:
- [intro](https://allennlp.org/elmo), [paper](https://www.aclweb.org/anthology/N18-1202), [quick tutorial](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md), [long tutorial](http://mlexplained.com/2019/01/30/an-in-depth-tutorial-to-allennlp-from-basics-to-elmo-and-bert/)


