import os
import pandas as pd
import numpy as np
from collections import defaultdict
from random import shuffle

def fifth_sent_stats(datasets):
    """ Get stats for 5th sentence in stories"""
    data_dicts = []
    # List of datasets
    for name, data in datasets.items():
        data_stats = defaultdict(int)
        data_stats['name'] = name

        # Loop stories in dataset
        for st_id, story in data.items():
            data_stats['story_count'] += 1

            # Placeholders
            maslow = []
            reiss = []
            motiv_text = []
            emotion_text = []
            emotion_plutchik = []

            # Get last line which we care about
            line_5 = story['lines']['5']

            # Log stats for character present
            for char, char_data in line_5['characters'].items():
                # Check if character appears
                if char_data["app"] == True:
                    # Motivation
                    for annotator, category in char_data["motiv"].items():
                        maybe_extend(maslow, category, 'maslow')
                        maybe_extend(reiss, category, 'reiss')
                        maybe_extend(motiv_text, category, 'text')
                    # Emotion
                    for annotator, emo in char_data["emotion"].items():
                        maybe_extend(emotion_text, emo, 'text')
                        maybe_extend(emotion_plutchik, emo, 'plutchik')

            if len(maslow) > 0:
                data_stats['motiv_maslow'] += 1
            if len(reiss) > 0:
                data_stats['motiv_reiss'] += 1
            if len(motiv_text) > 0:
                data_stats['motiv_text'] += 1
            if len(emotion_text) > 0:
                data_stats['emotion_text'] += 1
            if len(emotion_plutchik) > 0:
                data_stats['emotion_plutchik'] += 1
        data_dicts.append(data_stats)
    return data_dicts

def split_list_percent(ls, percent, shuff=True):
    """
    Return ls1 (size ls*(1-%)) and ls2 (size ls*%)
    """
    if shuff:
        shuffle(ls)
    n = int(len(ls)*percent)
    ls1 = ls[:-n]
    ls2 = ls[-n:]
    return ls1, ls2

def split_dict_percent(d, percent, shuff=True):
    """
    Return two dictionaries with `percent` being size of smaller dict
    """
    keys = list(d.keys())
    if shuff:
        shuffle(keys)
    n = int(len(keys)*percent)
    d1_keys = keys[:n]
    d2_keys = keys[-n:]
    d1 = {}
    d2 = {}
    for key, value in d.items():
        if key in d1_keys:
            d1[key] = value
        else:
            d2[key] = value
    return d1, d2

def dicts_to_pandas(d_list):
    keys = []
    names = []

    # Get all keys across all dictionaries
    for d in d_list:
        keys.extend(d.keys())
        names.append(d['name'])
    keys = set(keys)
    keys = sorted(keys)
    names = ['train', 'val', 'test']

    # Create dataframe
    df = pd.DataFrame(index=keys, columns=names)
    df = df.fillna(0) # with 0s rather than NaNs

    # Fill dataframe
    for d in d_list:
        name = d['name']
        for k, v in d.items():
            if 'name' not in k:
                df.loc[k].at[name] = v
    # del df['name']
    df = df.drop('name')
    return df

def pandas_to_markdown(df):
    # Dependent upon ipython
    # shamelessly stolen from https://stackoverflow.com/questions/33181846/programmatically-convert-pandas-dataframe-to-markdown-table
    from IPython.display import Markdown, display
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = pd.concat([df_fmt, df])
    #display(Markdown(df_formatted.to_csv(sep="|", index=False)))
    return Markdown(df_formatted.to_csv(sep="|", index=False))

def maybe_extend(ls, d, key):
    """
    Extends list `ls` with `d[key]` if key exists
    """
    if key in d:
        ls.extend(d[key])

def list_to_file(ls, path):
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        for line in ls:
            f.write(line+"\n")
    print(f"Saved data to file {path}")



