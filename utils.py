from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import (
    f1_score, recall_score, precision_score, 
    precision_recall_fscore_support, accuracy_score, 
    roc_auc_score, precision_recall_curve, average_precision_score
)
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from medspacy.custom_tokenizer import create_medspacy_tokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_pt_utils import get_parameter_names
from transformers import EvalPrediction, LlamaTokenizer
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from typing import Union
from pynvml import *
from torch import nn
import matplotlib.pyplot as plt
import scipy.stats as st
import dill as pickle
import bitsandbytes as bnb
import pandas as pd
import polars as pl
import numpy as np
import string
import scipy as sp
import torch
import transformers
import nlp
import shap
import re
import gc
import nltk
import datetime
import json
import collections
import operator
import spacy
import pynvml


# nltk.download('stopwords')

def preprocess_text(sentence: str) -> str:
    '''
    Preprocesses text data into simpler form (convert to lower case, remove numbers, special characters, punctuation, stopwords, etc)

    Args:
        sentence (str): a string that represents a sentence
    
    Returns:
        a simpler form of preprocessed text as a string
    '''
    lower_sentence = sentence.lower() # convert to lower case
    wo_hyphen = lower_sentence.replace('-', ' ')
    translator = str.maketrans('', '', string.punctuation) # remove punctuation
    wo_punc = wo_hyphen.translate(translator)
    units = ['cm', 'cc', 'beat', 'm2', 'l', 'gm', 's', 'min', 'mmol', 'hg', 'mm', 'ml', 'sq', 'm', 'sec', 'ms', 'cc/m2', 'gm/m2']
    wo_units = ' '.join(word for word in wo_punc.split() if word not in units) # remove units
    wo_space = ' '.join(wo_units.split()) # remove redundant spaces
    wo_num = re.sub(r'\d+', '', wo_space) # remove numbers
    wo_singlets = ' '.join([word for word in wo_num.split() if len(word) > 1]) # remove single characters
    wo_special_char = re.sub('[^A-Za-z0-9]+', ' ', wo_singlets) # remove special characters (all except alphanumeric chars)
    wo_white_space = " ".join(wo_special_char.split()) # remove white space
    stop_words = set(stopwords.words("english"))
    stop_words_to_remove = [
    'aren', "aren't", 'ain', 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'don', "don't",
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'nor', 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    ]
    for word in stop_words_to_remove:
        stop_words.remove(word)    
    word_tokens = word_tokenize(wo_white_space)
    wo_stop_words = [word for word in word_tokens if word not in stop_words] # remove stop words
    wo_stop_words = ' '.join(wo_stop_words)
    wo_single_chars =  re.sub(r"\b[a-zA-Z]\b", "", wo_stop_words) # remove single characters
    word_tokens = word_tokenize(wo_single_chars)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens] # lemmatization
    return ' '.join(lemmas)

def parse_line(text: str) -> list:
    '''
    parses unstructured text line by line and also remove empty strings.

    Args:
        text (str): a string of text data as a report or any kind of unstructured text data
    
    Returns: 
        a list of strings separated by lines
    '''
    return list(filter(bool, text.lower().strip().splitlines()))

def df_column2string(df: pd.DataFrame, col_name: str):
    '''
    go over each entry in a column and append the string as one single long string

    Args:
        df (pd.DataFrame): target DataFrame to go over
        col_name (str): a column name in df
    Returns: 
        a single long string with all the content in df['col_name'] concatenated
    '''
    assert col_name in df.columns, 'ValueError: col_name should be one of the column names in df!'
    assert not df.empty, 'ValueError: df must not be empty!'

    df.astype({col_name:'str'}).dtypes
    res_str, target_col_lis = '', list(df[col_name])

    for i in range(len(target_col_lis)):
        i_str = ' ' + preprocess_text(sentence=str(target_col_lis[i]))
        res_str += i_str

    return res_str    

def tokenize(input_text: str) -> list:
    preprocessed_input = preprocess_text(sentence=input_text)
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(preprocessed_input)]
    return tokenized_text

def read_model(path: str) -> nltk.lm.models.MLE:
    assert '.pkl' in path, 'ValueError: path name should be a .pkl file!'
    with open(path, 'rb') as fin:
        model_loaded = pickle.load(fin)
        return model_loaded

def clean_list(input_lis: list) -> list:
    cleaned_lis = [ele for ele in input_lis if ele != 'nan']
    cleaned_lis = [ele for ele in cleaned_lis if type(ele) == str]
    return cleaned_lis

def plot_wordcloud(input_text: str, max_words=100, **kwargs) -> None:
    '''
    plot a wordcloud image
    Args:
        input_text (str): a long single string as a corpus of text
        max_words (int): number of maximum number of words to be represented in the plot (default: 100)
        **kwargs: {filename: '.png'}: can store the .png file to a filename (str) path
    Returns: 
        None
    '''
    assert isinstance(max_words, int), 'ValueError: max_words should be type int!'

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=900, height=900, stopwords=stopwords, max_words=max_words, min_font_size=10).generate(input_text)

    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    if len(kwargs.items()) > 0:
        key = list(kwargs.keys())
        assert 'filename' in key, "ValueError: 'filename' should be one of the optional arguments"
        assert '.png' in kwargs['filename'], "ValueError: a filename must be a .png format"
        wordcloud.to_file(kwargs['filename'])

def string_to_dict(dict_string: str) -> dict:
    # Convert to proper json format
    dict_string = dict_string.replace("'", '"').replace('u"', '"')
    return json.loads(dict_string)

def string_to_list(list_string: str, remove_empty_str: bool, has_sqr_brkt: bool) -> list:
    # Convert string representation of list to a list (e.g. in simplified_multi_cmr_w_impr_meas.xlsx)
    if has_sqr_brkt:
        res_lis = list_string.strip('][').split(', ')
        res_lis = [ele.replace("'","") for ele in res_lis]
    else:
        res_lis = list(list_string.split(" "))
        
    if remove_empty_str:
        res_lis = [ele for ele in res_lis if ele]
    return res_lis

def get_frequency(input_lis: list, sort: bool) -> dict:
    '''
    get frequency of an unordered list of strings
    Args:
        input_lis (list): a list of strings
        sort (bool): whether to sort in a descending order by value
    
    Returns:
        a dictionary of keys (word) and values (frequency)
    '''
    frequency = dict(collections.Counter(input_lis))
    if sort:
        frequency = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    return frequency

def store_json(input_dict: dict, filename: str, **kwargs):
    json_obj = json.dumps(input_dict, indent=len(list(input_dict.keys())))

    append_str = ''
    if len(kwargs.items()) > 0: 
        append_lis = []
        for key, val in kwargs.items():
            append_lis.extend([str(key), str(val)])
        append_str += '_' + '_'.join(append_lis)
    
    with open(filename+append_str+'.json', "w") as outfile:
        outfile.write(json_obj)

def flatten_list(input_lis: list) -> list:
    return [item for sublist in input_lis for item in sublist]

def get_word_count(input_sent: str, preprocess: bool):
    '''
    get the number of words and unique words in a sentence
    
    Args:
        input_sent (str): a string representation of a string
        preprocess (bool): whether to preprocess the input text or not

    Returns:
        a count of words of the input (based on spaces) and unique word count
    '''
    if preprocess:
        input_sent = preprocess_text(sentence=input_sent)
    
    token = tokenize(input_text=input_sent)[0]
    
    return len(token), len(list(set(token)))
    
def text2abbrev(input_sent: str) -> str:
    file = open('models/abbreviations.json')
    json_data = json.load(file)
    json_keys = list(json_data.keys())
    res_str = input_sent
    for word in input_sent.split():
        if word in json_keys:
            if len(json_data[word]) == 1:
                res_str.replace(word, json_data[word][0])
            elif len(json_data[word]) > 1:
                count_lis = [input_sent.count(candidate_word) for candidate_word in json_data[word]]
                max_idx = count_lis.index(max(count_lis))
                res_str.replace(word, json_data[word][max_idx])
    
    return res_str

def get_elapsed_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded))

def preprocess_impr_df(report_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove rows with 'impr_sent' with one one less words
    '''
    res_df = report_df[report_df['impr_sent'].apply(lambda x: len(x.split()) > 1)]
    return res_df

def clean_text(text: str) -> str:
    '''
    from /data/aiiih/projects/huangp2/ccf/utils.py
    '''
    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS_TO_REMOVE = [ 'aren',
                        "aren't",
                        'ain',
                        'couldn',
                        "couldn't",
                        'didn',
                        "didn't",
                        'doesn',
                        "doesn't",
                        'don',
                        "don't",
                        'hadn',
                        "hadn't",
                        'hasn',
                        "hasn't",
                        'haven',
                        "haven't",
                        'isn',
                        "isn't",
                        'mightn',
                        "mightn't",
                        'mustn',
                        "mustn't",
                        'needn',
                        "needn't",
                        'no',
                        'nor',
                        'not',
                        'shan',
                        "shan't",
                        'shouldn',
                        "shouldn't",
                        'wasn',
                        "wasn't",
                        'weren',
                        "weren't",
                        'won',
                        "won't",
                        'wouldn',
                        "wouldn't",
                        ]
    for word in STOPWORDS_TO_REMOVE:
        STOPWORDS.remove(word)

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    UNITS = {'cm', 'cc', 'beat', 'm2', 'l', 'gm', 's', 'min', 'mmol', 'hg', 'mm', 'ml', 'sq', 'm', 'sec', 'ms'}
    QUANTS = {'edv', 'edvi', 'esv', 'esvi', 'svi', 'lvef',  'lvmi', 'rvef', 'ef', 'sv', 'co', 'ci', 'ce'}
    QUANTS_PHRASE = {'stroke volume', 'cardiac index', 'lv mass', 'cardiac output', 'forward volume', 
                    'reverse volume', 'net forward volume', 'aortic regurgitant fraction', 'quantitative mitral regurgitant volume',
                    'quantitative mitral regurgitant fraction'}
    
    text = text.lower() # lowercase text
    text = text.replace('result:', '')
    text = text.replace('impression:', '')
    text = text.replace('overall', '')
    text = text.replace('\n', ' ') # remove new line
    text = text.replace('\t', ' ') # remove new line
    text = re.sub("[\(\[].*?[\)\]]", "", text) # remove texts () and []

    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = ' '.join(word for word in text.split() if word not in UNITS) # remove units from text
    text = ' '.join(word for word in text.split() if word not in QUANTS) # remove quantitative measures from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwords from text

    for phrase in QUANTS_PHRASE:
        text = text.replace(phrase, ' ')

    text = re.sub(r'[0-9]', '', text) # remove digits
    text = ' '.join(text.split()) # remove redundant spaces

    return str(text)

def load_npy_image(img_path: str) -> np.array:
    image_np = np.load(img_path)
    image_tensor = torch.from_numpy(image_np)

    return image_tensor

    
def configure_loss(target, device):
    # Get class importance weights
    class_sample_count = np.unique(target, return_counts=True)[1]
    weight = 1. / class_sample_count
    class_weights = weight / weight.sum()
    loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).float().to(device))

    return loss_func

def sort_report_by(df: pl.DataFrame, by: list, agg: str, sort_by: list, select: list, collapse=True) -> pl.DataFrame:
    '''
    returns a dataframe with aggregated column grouped by other columns.
    Args:
        df (pl.DataFrame): dataframe to process
        by (list): a list of columns to group by
        agg (str): a column to aggregate
        sort_by (list): a column to sort
        select (list): a list to selected for the result dataframe
        collapse (bool): whether or not to collapse the report by line in 'CONCAT_NOTE_TEXT'
    Returns:
        a new dataframe aggregated and grouped by arguments listed above.
    '''
    df = df.drop_nulls() # drop rows that have None-types for all columns (by default with no arguments)
    df = df.sort(sort_by)
    df = df.to_pandas()
    if collapse:
        df['CONCAT_NOTE_TEXT'] = df.groupby(by)[agg].transform(lambda x : '\n'.join(x))
    else:
        df['CONCAT_NOTE_TEXT'] = df[agg]
    return pl.from_pandas(df).unique(subset=['K_PAT_KEY']).select(select)

def cut_words(text: str, max_len: int) -> str:
    '''
    cut words in a text until the max_length.
    Args:
        text (str): unstructured text
        max_len (int): maximum number of words 
    Returns:
        a new text that has maximum length of max_len
    '''
    words = text.split()
    if len(words) <= 512:
        return text
    else:
        reduced_text = ' '.join(words[:max_len])
        return reduced_text
    
def word_count(text):
    '''
    Return number of words in a string
    '''
    return len(text.split())

def remove_names_with_prefix(text):
    pattern = r'dr\.\s+[A-Z][a-z]+'
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return cleaned_text

def remove_names(report):
    '''
    remove names and any organization
    Args:
        report (str): an unstructured report.
    Returns:
        a report without any names/organization
    '''
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(report)
    ents = [e.text.split('(')[0] for e in doc.ents] # split '(' and leave the first part since it can cause escape char issues
    
    if len(ents) > 1:
        ents_pattern = r'|'.join(ents)
        return re.sub(ents_pattern, '', report)
    elif len(ents) == 1:
        return re.sub(ents[0], '', report)
    else:
        return report


def remove_measurement_values(report: str) -> str:
    '''
    remove measurement values (numerical value and units)
    Args:
        report (str): an unstructured report.
    Returns:
        a report without any measurement values
    '''
    units = '(c)?m|cc|bpm|beat|m2|l|gm|oz|(mc|m|c)?g|s|min|(m)?molhg|(m)?m(Hg)?|(m)?l|sq|degree(s)?|hour(s)?|time(s)?|tablet(s)?|lb|kg|%'
    pattern = r'\b\d+(\.\d+)?\s*({})'.format(units)
    return re.sub(pattern, '', report)

def remove_time_info(report: str) -> str:
    '''
    remove any time information (i.e. year, month, day, time)
    Args:
        report (str): an unstructured report.
    Returns:
        a report without any time information
    '''
    month_pattern = r'january|february|march|april|may|june|july|august|september|october|november|december'
    date_pattern = r'\d{1,2}(/|:)\d{1,2}(/|:)\d{4}|\d{4}-\d{2}-\d{2}'
    time_pattern = r'\d{1,2}:\d{2}(?:am|pm|AM|PM)?'
    ampm_pattern = r'am|pm|AM|PM'

    wo_month = re.sub(month_pattern, '', report, flags=re.IGNORECASE)
    wo_date = re.sub(date_pattern, '', wo_month)
    wo_time = re.sub(time_pattern, '', wo_date)
    return re.sub(ampm_pattern, '', wo_time)

def remove_punctuation(report_lis: list) -> list:
    assert type(report_lis) == list, 'This function requires input type of list!'
    translator = str.maketrans("", "", string.punctuation)
    return [report.translate(translator) for report in report_lis]

def remove_nondiagnostic_info(report: str, _remove_names: bool) -> str:
    '''
    remove any nondiagnostic information in an unstructured report
    Args:
        report (str): an unstructured report.
        _remove_names (bool): whether to remove named entities using spacy
    Returns:
        a report with only diagnostic information
    '''
    # use helper function to pre-preprocess report
    report = remove_measurement_values(report=report)
    report = remove_time_info(report=report)

    STOPWORDS = set(stopwords.words('english'))
    STOPWORDS.update([
        'mr', 'dr', 'md', 'surgeon', 'asst', 'assistant', 'physician', 'result:',
        'impression:', 'overall', 'ne', 'oh', 'mrn', 'cleveland', 'ohio', 'canton',
        'clinic'
    ])

    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;_]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_.]')
    UNITS = {'cm', 'cc', 'beat', 'm2', 'l', 'gm', 's', 'min', 'mmol', 'hg', 'mm', 'mc', 'ml', 'sq', 'm', 'sec', 'ms', 'mhg'}
    QUANTS = {'edv', 'edvi', 'esv', 'esvi', 'svi', 'lvef',  'lvmi', 'rvef', 'ef', 'sv', 'co', 'ci', 'ce'}
    QUANTS_PHRASE = {'stroke volume', 'cardiac index', 'lv mass', 'cardiac output', 'forward volume', 
                    'reverse volume', 'net forward volume', 'aortic regurgitant fraction', 'quantitative mitral regurgitant volume',
                    'quantitative mitral regurgitant fraction'}
    
    report = report.replace('\n', ' ') # remove new line
    report = report.replace('\t', ' ') # remove new line
    report = re.sub("[\(\[].*?[\)\]]", "", report) # remove texts in () and []
    report = re.sub(r'[^.,a-zA-Z ]', '', report) # remove special characters except a dot
    report = re.sub(r'\([^)]*\)|\[[^]]*\]|\{[^}]*\}', '', report) # remove brackets/parentheses
    report = re.sub(r'(?<=[a-zA-Z])\.(?!\s|$)', '', report) # remove dots that are not followed by a space or end of line.
    if _remove_names:
        report = remove_names(report=report) # helper function that remove names 
    report = report.lower() # lowercase text

    report = REPLACE_BY_SPACE_RE.sub(' ', report) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    report = BAD_SYMBOLS_RE.sub('', report) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    report = ' '.join(word for word in report.split() if word not in UNITS) # remove units from text
    report = ' '.join(word for word in report.split() if word not in QUANTS) # remove quantitative measures from text
    report = ' '.join(word for word in report.split() if word not in STOPWORDS) # remove stopwords from text

    for phrase in QUANTS_PHRASE:
        report = report.replace(phrase, ' ')

    history_match = re.search(r'(?i)(?<=history).*', report)
    if history_match:
        report = history_match.group().strip()
    
    report = re.sub(r'(physician|contact information).*?(fax|oh|ohio)', '', report) # remove contact information sections
    report = re.sub(r'heart vascular institute.*?ohio', '', report) # remove address sections
    report = re.sub(r'\b\d+([./]\d+)+\b', '', report) # remove digits part 1
    report = re.sub(r'[0-9]', '', report) # remove digits part 2
    report = re.sub(r'cleveland clinic( foundation)? euclid avenue cleveland oh(io)? usa (epic care )?(operative report )?', '', report) # remove hospital name and address
    report = re.sub(r'\b\w\b', '', report) # remove any single digit words
    report = ' '.join(report.split()) # remove redundant spaces

    return str(report)


def str2bool_label(df: pl.DataFrame, cols: list) -> pl.DataFrame:
    '''
    convert string representation (e.g. 'Yes', 'No', 'Null') to 0 and 1
    Args: 
        df (pl.DataFrame): input dataframe that has label columns, such as 'SP_' columns.
        cols (list): input list to change data type from string to int
    Returns:
        a dataframe with lables represented in int 0 or 1.
    '''
    pos_label_lis = ['yes', 'Yes', 'YES']
    neg_label_lis = ['no', 'No', 'NO', 'null', 'Null', 'NULL', 'None']
    filter_lis = pos_label_lis + neg_label_lis

    for col in cols:
        df = df.filter(pl.col(col).str.contains('|'.join(filter_lis)))
        df = df.with_columns(
            pl.col(col).str.replace(r"{}".format('|'.join(pos_label_lis)), 1).str.replace(r"{}".format('|'.join(neg_label_lis)), 0)
        )
        df = df.with_columns(pl.col(col).cast(pl.Int64))
    return df

def get_gpu_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, average: str):
    y_true = y_true.cuda()
    y_pred = y_pred.cuda()

    if average == 'micro':
        tp = torch.sum(y_true * y_pred)
        fp = torch.sum((1 - y_true) * y_pred)
        fn = torch.sum(y_true * (1 - y_pred))
    else: # macro and weighted
        tp = torch.sum(y_true * y_pred, dim=0)
        fp = torch.sum((1 - y_true) * y_pred, dim=0)
        fn = torch.sum(y_true * (1 - y_pred), dim=0)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    if average == 'micro':
        return 2 * precision * recall / (precision + recall + 1e-8)
    elif average == 'macro':
        return torch.mean(2 * precision * recall / (precision + recall + 1e-8))
    else: # weighted
        f1_weighted = 2 * precision * recall / (precision + recall + 1e-8)
        return torch.mean(torch.sum(f1_weighted * torch.sum(y_true, dim=0) / torch.sum(y_true)))



def get_gpu_auc(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.cuda()
    y_pred = y_pred.cuda()

    sorted_indices = torch.argsort(y_pred, descending=True)
    y_true_sorted = y_true[sorted_indices]
    cum_sum = torch.cumsum(y_true_sorted, dim=0)
    threshold_idxs = torch.where(y_true_sorted[1:] != y_true_sorted[:-1])[0] + 1
    fp = torch.cat([torch.tensor([0]).cuda(), cum_sum[threshold_idxs[1:] - 1]])
    tp = torch.cat([torch.tensor([0]).cuda(), cum_sum[threshold_idxs[:-1]]])
    fpr = fp / fp[-1]
    tpr = tp / tp[-1]
    return torch.trapz(tpr, fpr)

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            }

def optimal_threshold_precision_recall_curve(preds, precision, recall, thresholds):
        preds, precision, recall, thresholds = np.array(preds), np.array(precision), np.array(recall), np.array(thresholds)
        optimal_thresholds = sorted(list(zip(np.abs(precision - recall), thresholds)), key=lambda i: i[0], reverse=False)[0][1]
        optimal_mask = np.where(preds>optimal_thresholds,1,0)
        return optimal_thresholds, optimal_mask

def get_pos_lis(df: pl.DataFrame, label_lis: list, min_pos_rate: float, ignore_labels=True):
    count, pos_lis = 0, []
    for label in label_lis:
        try:
            pos_num = df.groupby(label, maintain_order=True).agg(pl.count()).filter(pl.col(label) == 1)[:, 1].item()
            pos_rate = pos_num / len(df)
        except:
            pos_rate = 0.0
        
        
        if pos_rate > min_pos_rate:
            count +=1
            pos_lis.append(label)
    
    if ignore_labels:
        pos_set = set(pos_lis)
        labels2ignore = set([
        "SP_PV_Repair", "SP_Valve_Sparing_Root_remodeling_Yacoub", "SP_Valve_Sparing_Root_Reconstruction_Florida",
        "SP_AV_Ring_Annuloplasty", "SP_AV_Leaflet_Free_Edge_Reinforcement", "SP_AV_Leaflet_Pericardial_Patch",
        "SP_AV_Division_of_Fused_Leaflet_Raphe", "SP_AV_Repair_of_Periprosthetic_Leak", "SP_AV_Implant_Type_Autograft_Ross",
        "SP_MV_Implant_Type_Other", "SP_MV_Chords_Preserved_Type", "SP_Transmyocardial_Laser_Revascularization",
        "SP_Trauma_Repair", "SP_Subaortic_Stenosis_Resection", "SP_TV_Explant", "CB_Gastroepiploic_Inf_Epigastric_Graft_To_LAD",
        "CB_Gastroepiploic_Inf_Epigastric_Graft_To_Diagonal", "CB_Gastroepiploic_Inf_Epigastric_Graft_To_LCX",
        "CB_Other_Conduit_Procedure_To_RCA", "CB_Other_Conduit_Procedure_To_LAD", "CB_Other_Conduit_Procedure_To_Diagonal",
        "CB_Other_Conduit_Procedure_To_LCX", "SP_MV_Transcatheter_Replacement", "SP_MV_Leaflet_Clip", "SP_Cath_Based_Assist_Device_Used",
        "SP_MV_Native_Pannus_Thrombus_Removal", "SP_MV_Prosthetic_Valve_Repair", "SP_AV_Exploration", "SP_MV_Exploration",
        "SP_PV_Exploration", "SP_TV_Exploration", "Preop_Lab_Hematocrit", "HX_TV_Replacement", "Cardiac_Symptoms_on_Admission", "Preop_Resuscitation"
        ])
        pos_lis = list(pos_set-labels2ignore)

    return count, pos_lis

def focal_binary_cross_entropy(logits, targets, num_label, gamma=2):
    '''
    Reference: https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained
    '''
    p = torch.where(targets >= 0.5, logits, 1-logits)
    logp = - torch.log(torch.clamp(logits, 1e-4, 1-1e-4))
    loss = logp*((1-logits)**gamma)
    loss = torch.mean(input=loss, dim=1)*num_label
    return loss.mean()

def compute_precision_recall_f1(predictions, targets):
    # Compute true positives, false positives, and false negatives
    true_positives = torch.sum((predictions == 1) & (targets == 1)).float()
    false_positives = torch.sum((predictions == 1) & (targets == 0)).float()
    false_negatives = torch.sum((predictions == 0) & (targets == 1)).float()

    # Compute precision
    try:
        precision = true_positives / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0
        print('PRECISION COMPUTATION INCLUDES A ZERO DIVISION!')

    # Compute recall
    try:
        recall = true_positives / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0.0
        print('RECALL COMPUTATION INCLUDES A ZERO DIVISION!')

    # Compute F1 score
    try:
        f1_score = 2 * ((precision * recall) / (precision + recall))
    except:
        f1_score = 0.0
        print('F1 SCORE COMPUTATION INCLUDES A ZERO DIVISION!')

    return precision.item(), recall.item(), f1_score.item()

def extract_num_from_str(input_str: str) -> int:
    return re.findall('\d+', input_str)[0]

def sectionize_report(report: str) -> dict:
    '''
    get sections that are capital letters only following a ':'
    Args:
        report (str): an unstructured report
    Returns:
        a dictionary of sections and its content
    '''
    pattern = r'(?<!\w)[A-Z][A-Z\s]*:'
    pattern = r'((?:\b\w+\s+){0,2}\b\w+)\s*:'
    try: 
        sections = re.split(pattern, report)
        sections = [section.strip() for section in sections if section.strip()]
        headers = re.findall(pattern, report)
        sections_with_contents = {}
        for i, ele in enumerate(headers):
            if i < len(headers) -1:
                match = re.search('{}(.*?){}'.format(ele, headers[i + 1]), report).group(1) # Adding a ? on a quantifier (?, * or +) makes it non-greedy
                sections_with_contents[ele] = match
            else:
                match = re.search('{}(.*?)'.format(ele), report).group(1)
                sections_with_contents[ele] = match

        return sections_with_contents
    except:
        return dict()

def get_medical_history_sections(report: str) -> str:
    '''
    get medical history sections concatenated together
    Args:
        report (str): an unstructured report
    Returns:
        a concatenated string of medical histories/comorbidities
    '''
    extracted_sections = {} # Initialize a dictionary to store extracted text segments
    header_pattern = r'([A-Z\s]+:)\s*$' # Define the regular expression pattern for section headers
    hpi_pattern = r'(PMH(x)?|Hx|PSH|HPI|(History of )?Present Illness|history of|IMPRESSION|Problem List|Active Major Problems|Comorbidities|Principal Diagnosis|PAST (MEDICAL|SURGICAL) HISTORY)((\s*)?:)?' # Define the regular expression pattern
    sections = re.split(hpi_pattern, report, flags=re.MULTILINE | re.IGNORECASE) # Split the text into sections based on section headers

    current_section_header, current_section_content = None, ''

    try:
        sections = list(filter(None, sections))
        sections_bool = [bool(re.search(hpi_pattern, section)) for section in sections]
        first_true_idx = sections_bool.index(True) # first occurence of the header
        sections, sections_bool = sections[first_true_idx+1:], sections_bool[first_true_idx+1:]
        true_idx = [i for i, x in enumerate(sections_bool) if not x]

        return ' '.join([sections[i] for i in true_idx])
    except:
        return report

    # # Iterate through the sections
    # for section in sections:
    #     if re.match(hpi_pattern, section.strip()):
    #         # We've found a new HPI or PAST MEDICAL HISTORY section
    #         if current_section_header is not None:
    #             # Store the previous section
    #             extracted_sections[current_section_header] = current_section_content.strip()
            
    #         # Initialize the new section
    #         current_section_header = section.strip(':').strip()
    #         current_section_content = ''
    #     else:
    #         # Append content to the current section
    #         current_section_content += section
    
    # # Store the last section
    # if current_section_header is not None:
    #     extracted_sections[current_section_header] = current_section_content.strip()
    
    # return extracted_sections

def get_procedure_after(report: str) -> str:
    report = re.split(r"\*+", report) # multiple report separator
    report = [re.split(r'procedure:|\s+operative report(:)?\s+', report_i, maxsplit=1, flags=re.IGNORECASE) for report_i in report]
    report = [report_i for report_i in report if len(report_i) > 1] # remove reports that don't have 'PROCEDURE:'
    report = [list(filter(None, report_i)) for report_i in report]
    try:
        report = '\n'.join(['\n'.join(report_i[1:]) for report_i in report]) # removed first part is before 'PROCEDURE:'
    except:
        report = None
    return report

def initialize_medspacy_nlp_pipeline():
    '''
    Helper function for sentence_splitter() since initializing the nlp pipieline everytime in 
    sentence_splitter() takes too much time.
    '''
    nlp = spacy.blank("en")
    medspacy_tokenizer = create_medspacy_tokenizer(nlp)
    nlp.tokenizer = medspacy_tokenizer
    nlp.add_pipe("medspacy_pyrush")

    return nlp

def sentence_splitter(report: str, min_len: int, nlp) -> list:
    '''
    Splits a report into sentences based on medsapcy algorithm.
    Args:
        report (str): unstructured report. 
        min_len (int): minimum number of words in a sentence. If sentence length is less than min_len, delete sentence from result list.
        nlp : initialized medspacy nlp pipeline
    Returns:
        a segmented report in a list of sentences.
    '''
    doc = nlp(report)
    res_lis = [sent.text for sent in doc.sents]
    res_lis = [' '.join([res_lis[i-1], res_lis[i]]) if i > 0 and len(ele.split()) < min_len else ele for i, ele in enumerate(res_lis)]

    return res_lis

def get_procedure_section(report: str) -> str:
    primary_start_words = re.compile(r'''
    (OPERATIVE|SURGERY\/)?PROCEDURE(:)?|OPERATIVE REPORT(:)?|
    DESCRIPTION OF PROCEDURE(:)?|PROCEDURE (IN )?DETAIL(S)?(:)?|OPERATION(S)?(:)?|
    (Operative|Surgery\/)?Procedure:|Operative Report:|
    Description (Of|of) Procedure:|Procedure (In |in )?Detail(s)?:|Operation(s)?:''', re.X)
    secondary_start_words = re.compile(r'''
    NAME OF OPERATION(:)?|OPERATIVE (FINDINGS|INDICATIONS)(:)?|
    FINDINGS AND TECHNIQUE(:)?|PROCEDURE TYPE(:)?|
    Name (Of|of) Operation:|Operative (Findings|Indications):|
    Findings (And|and) Technique:|Procedure (T|t)ype:''', re.X)
    end_words = re.compile(r'\b\w+:|\*+|\#:', flags=re.IGNORECASE) 
    final_res_proc = ''

    # find primary and secondary word matches
    start_word_search, second_start_word_search = re.search(primary_start_words, report), re.search(secondary_start_words, report)
    while bool(start_word_search) or bool(second_start_word_search): # iterate until there're no start words found
        res_report, start_idx = report, 0

        # finding the starting point of the extracted report
        if bool(start_word_search):
            start_idx = start_word_search.end()
        elif bool(second_start_word_search):
            start_idx = second_start_word_search.end()
        else:
            break

        # cut out words that are before the start word
        res_report, report = report[start_idx:], report[start_idx:] 

        # finding the ending point of the extracted report
        end_word_search = re.search(end_words, res_report)
        if bool(end_word_search) and (bool(start_word_search) or bool(second_start_word_search)):
            end_idx = end_word_search.start()
            res_report, report = res_report[:end_idx-1], report[end_idx:] 
            final_res_proc += ' ' + res_report 
            start_word_search, second_start_word_search = re.search(primary_start_words, report), re.search(secondary_start_words, report)
        else: # doesn't have an end word 
            final_res_proc += ' ' + res_report
            break

    return final_res_proc

def remove_sections_from_report(report: str) -> str:
    '''
    remove all the unnecssary and useless sections that exist in a report
    Args:
        report (str): an unstructure report
    Returns:
        a report that removes all the unnecessary sections and its corresponding contents
    '''
    sections2remove = ['MEDICAL CENTER', 'DATE', 'CC', 'PRIMARY CARE', 'DT', 'DD', 'SURGEON']
    
    sections_with_contents = sectionize_report(report=report)
    try: # sections_with_contents might be null
        for k, v in sections_with_contents.items():
            report = re.sub(r'{}|{}'.format(k, v), '', report)
    except:
        pass

    return report

def report_gpu():
    try:
        print('BEFORE CLEARING CUDA CACHE', torch.cuda.list_gpu_processes())
        torch.cuda.empty_cache() # get your GPU back to a clean slate of not using more memory than it need to
        gc.collect() # call python's garbage collection
        print('AFTER CLEARING CUDA CACHE', torch.cuda.list_gpu_processes())
    except:
        print('GPU NOT FOUND!')

def get_avail_gpu():
    try:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print('TOTAL MEMORY: {}, RESERVED MEMORY: {}, ALLOCATED MEMORY: {}, FREE MEMORY INSIDE RESERVED: {}'.format(t, r, a, f))
    except:
        print('GPU NOT FOUND!')

def get_loss_weights(df: pl.DataFrame, col_lis: list, how='non_null_reinforcement') -> torch.tensor:
    '''
    get weights for a loss function
    Args:
        df (pl.DataFrame): original data in dataframe format
        col_lis (list): list of columns of class labels
    Returns:
        a tensor of size len(col_lis) that contains class weights for a loss function
    '''
    res_lis = []
    for col in col_lis:
        if how == 'non_null_reinforcement':
            try:
                null_count = df[col].value_counts().filter(pl.col(col) == None)['counts'].item()
            except:
                null_count = int(np.sqrt(len(df)))
                
            res_lis.append(1.0-((len(df) - null_count) / len(df)))
        elif how == 'positive_example_reinforcement':
            try:
                pos_count = df[col].value_counts().filter(pl.col(col) == 1)['counts'].item()
            except:
                pos_count = int(np.sqrt(len(df)))
                
            res_lis.append(1.0 - (pos_count / len(df)))

    lower, upper = 0.05, 0.95
    res_lis = [lower + (upper - lower) * ele for ele in res_lis] # normalize to [lower, upper]
    return torch.tensor(res_lis, dtype=torch.float16) 

def flatten_tokens(token: BatchEncoding, max_length: int) -> BatchEncoding:
    '''
    Remove tokens in the middle where [cls] is repeated
    Args:
        token (BatchEncoding): a dictionary of keys 'input_ids', and 'attention_mask'
        max_length (int): maximum length of a token
    Returns:
        a cleaned up token flattened and cleaned up [cls] tokens
    '''
    # flatten out list of lists to a single list
    token['input_ids'] = [ele for sublist in token['input_ids'] for ele in sublist]
    token['attention_mask'] = [ele for sublist in token['attention_mask'] for ele in sublist]

    for i, ele in enumerate(token['input_ids']): # remove [cls] tokens in the middle
        if i != 0 and ele == 2:
            del token['input_ids'][i]
            del token['attention_mask'][i]

    if len(token['input_ids']) >= max_length: # remove over-sized list
        token['input_ids'], token['attention_mask'] = token['input_ids'][:max_length-1], token['attention_mask'][:max_length-1]
        token['input_ids'].append(3) # append a [sep] token at the end
        token['attention_mask'].append(1)
    else: # we need to pad to max_length
        token['input_ids'].extend([1]*(max_length - len(token['input_ids'])))
        token['attention_mask'].extend([0]*(max_length - len(token['attention_mask'])))

    
    return token

def get_latest_checkpoint_file(directory_path: str, substring: str) -> str:
    '''
    get the latest file updated in a folder with a filename that has substring "checkpoint"
    Args:
        directory_path (str): path that the file resides in
        substring (str): substring of the complete filename
    Returns:
        the full filename of the latest checkpoint file
    '''
    subdirectories = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f)) and substring in f]
    latest_subdirectory = max(subdirectories, key=lambda d: os.path.getmtime(os.path.join(directory_path, d)))
    return latest_subdirectory

def group_texts(examples):
    block_size = 512
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def initialize_8bit_adam(training_args, model):
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }
    optimizer_kwargs["lr"] = training_args.learning_rate
    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )

    return adam_bnb_optim

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    pynvml.print_gpu_utilization()

def print_gpu_utilization():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def count_parameters(model):
    try:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_portion = pytorch_trainable_params / pytorch_trainable_params

        print('Total parameters: {} || total trainable parameters: {} || trainable%: {}'.format(f'{pytorch_total_params:,}', f'{pytorch_trainable_params:,}', trainable_portion))
    
    except:
        print('UNCOUNTABLE PARAMETERS...')


def compute_metrics(pred: EvalPrediction):
    def multi_label_metrics(predictions, labels, threshold=0.5):
        '''
        helper function for compute_metrics():
        # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
        '''
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
        prec_micro, rec_micro, _, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='micro')

        # calculate individual label f1 and AUCs
        auc_scores, f1_scores = [], []
        for i in range(labels.shape[1]): # loop through number of labels
            preds_i, labels_i = predictions[:, i], labels[:, i]
            preds_i = sigmoid(torch.Tensor(preds_i))
            auc_scores_i = roc_auc_score(labels_i, preds_i)
            precision, recall, thresholds = precision_recall_curve(y_true=labels_i, probas_pred=preds_i)
            optimal_thresholds, preds_binary_i = optimal_threshold_precision_recall_curve(preds=preds_i, precision=precision, recall=recall, thresholds=thresholds)
            # preds_binary_i = torch.where(preds_i >= threshold_val, 1, 0) # change to binary predictions from probabilities
            precision_i, recall_i, f1_scores_i = compute_precision_recall_f1(predictions=torch.tensor(preds_binary_i), targets=labels_i)
            auc_scores.append(auc_scores_i)
            f1_scores.append(f1_scores_i)


        # return as dictionary
        metrics = {
            'f1_micro_average': f1_micro_average, 'roc_auc_micro': roc_auc, 'precision_micro': prec_micro, 'recall_micro': rec_micro
            }    
        metrics.update({'f1_label{}'.format(n): score for n, score in enumerate(f1_scores)})
        metrics.update({'auc_label{}'.format(n): score for n, score in enumerate(auc_scores)})

        return metrics
    
    preds = pred.predictions[0] if isinstance(pred.predictions, 
            tuple) else pred.predictions
    # breakpoint()
    result = multi_label_metrics(
        predictions=preds, 
        labels=pred.label_ids)
    
    return result

def tokenize(examples):
    '''
    used for mapping datset of raw text to tokenized inputs using llama tokenizer
    '''
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token # (0)
    text = examples['CONCAT_NOTE_TEXT'] # take a batch of texts
    # breakpoint()
    # nlp = initialize_medspacy_nlp_pipeline()
    # text = [remove_punctuation(sentence_splitter(report=text_i, min_len=3, nlp=nlp)) for text_i in text]
    label_names = [label for label in examples.keys() if label != 'CONCAT_NOTE_TEXT']
    encoding = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_token_type_ids=False) # encode the text field
    # encoding = [flatten_tokens(token=encoding_i, max_length=512) for encoding_i in encoding]
    labels_batch = {k: examples[k] for k in examples.keys() if k in label_names} # add labels separately from text
    labels_matrix = np.zeros((len(text), len(label_names))) # create numpy array of shape (batch_size, num_labels)
    for idx, label in enumerate(label_names): # fill numpy array
        labels_matrix[:, idx] = labels_batch[label]
    encoding['labels'] = labels_matrix.tolist()

    return encoding

def calculate_ci(x: list, only_ci=True):
    '''
    Reference: https://rowannicholls.github.io/python/statistics/confidence_intervals.html
    #:~:text=Confidence%20Interval%20of%20the%20Mean%20of%20a%20Small%20Sample,
    -Now%2C%20having%20an&text=In%20other%20words%2C%20there%20is,x%E2%88%92CI%3C%CE%BC)
    '''
    x = np.array(x)
    x = x[~np.isnan(x)] # remove all NaN values from np.array
    x_bar = np.mean(x) # sample mean
    s = np.std(x, ddof=1) # use ddof=1 to get the sample standard distribution
    n = len(x) # sample size
    cl = 0.95 # confidence level
    alpha = 1 - cl # significance level 
    tails = 2 # number of tails
    q = 1 - (alpha / tails) # quantile (the cumulative probability)
    dof = n - 1 # degrees of freedom
    t_star = st.t.ppf(q, dof) # critical t-statistic, calculated using the percent-point function of the t-distribution
    confidence = t_star * s / np.sqrt(n)
    ci_upper, ci_lower = x_bar + confidence, x_bar - confidence # confidence interval
    
    if only_ci:
        return confidence
    else:
        return x_bar, confidence, [ci_lower, ci_upper]

def visualize_shap(model_name: str, data: pl.DataFrame, col_name: str):
    import pyarrow as pa
    import pyarrow.dataset as ds
    import shap

    # load a BERT sentiment analysis model
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(model_name)
    model = transformers.DistilBertForSequenceClassification.from_pretrained(model_name).cuda()

    data = data.select(
        [
            col for col in data.columns if 'SP_' in col or col=='CONCAT_NOTE_TEXT'
        ]
    )
    data = data.select(
        ['CONCAT_NOTE_TEXT', col_name]
    ).rename({'CONCAT_NOTE_TEXT': 'text', col_name: 'label'}).drop_nulls()
    first_pos_idx = data['label'].to_list().index(1.0)

    # define a prediction function
    def f(x):
        tv = torch.tensor(
            [
                tokenizer.encode(v, padding="max_length", max_length=512, truncation=True)
                for v in x
            ]
        ).cuda()
        outputs = model(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
        return val

    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer)
    shap_values = explainer(data[:first_pos_idx+1].to_dict(), fixed_context=1)

    print(data[first_pos_idx, 'text'])
    shap.plots.text(shap_values[first_pos_idx])
    # plt.savefig('figures/shap_summary_{}.png'.format(col_name), dpi=300)

def resize_dpi(img_path: str, dpi: int):
    '''
    Resize an image into a desired dpi
    Args:
        img_path (str): file path to the image of image file type (e.g. jpg, png, svg, etc)
        dpi (int): integer value for dots per inches (typically >= 300 for publications)\
    Returns: 
        None
    '''
    from PIL import Image

    im = Image.open(img_path)
    im.save('{}_{}'.format(dpi, img_path), dpi=(dpi, dpi))

def get_false_predictions(tokenizer, sent_ids, total_preds_binary, total_labels):
    def get_pure_decoded_str(sent_ids):
        decoded_sents = tokenizer.decode(sent_ids[i, :] for i in range(sent_ids.shape[0]))
        decoded_sents = [sent.split('[unused1]', 1)[0] for sent in decoded_sents]
        return decoded_sents
    
    decoded_sents = get_pure_decoded_str(sent_ids=sent_ids)
    total_preds_binary, total_labels = np.array(total_preds_binary), np.array(total_labels)
    chr_arr = np.chararray((total_labels.shape[0], total_labels.shape[1]))
    for i in range(total_labels.shape[0]):
        for j in range(total_labels.shape[1]):
            if total_preds_binary[i, j] == 1 and total_labels[i, j] == 0: # FP case
                chr_arr[i, j] = 'fp'
            elif total_preds_binary[i, j] == 0 and total_labels[i, j] == 1: # FN case
                chr_arr[i, j] = 'fn'
            elif total_preds_binary[i, j] == 1 and total_labels[i, j] == 1: # TP case
                chr_arr[i, j] = 'tp'
            else:
                chr_arr[i, j] = 'tn'
    
    res_df = pl.from_numpy(chr_arr).with_columns(
        pl.Series(name='decoded_sents', values=decoded_sents)
    )
    return res_df
    
def infer_from_pretrained(model_path: str, label_names: list, test_dataloader, bert_model_name, batch_size, device):
    from cv_classifier import BertClassifier
    model = BertClassifier(bert_model_name=bert_model_name, n_labels=len(label_names), unfreeze=0.10).to(device) # load a model which is a torch.nn.Module object 
    model.load_state_dict(torch.load(model_path)) # load state file that contains parameters
    model.to(device)
    model.eval()

    total_preds = torch.empty((batch_size, len(label_names)), dtype=torch.half)
    for step, batch in enumerate(test_dataloader):
        batch = [val.to(device) if key != 'labels' else val for key, val in batch.items()]
        sent_id, mask, labels = batch
        with torch.inference_mode():
            torch.cuda.empty_cache()
            preds = model(sent_id, mask)

            if step == 0:
                total_preds = preds[:, :len(label_names)]
            else:
                total_preds = torch.vstack((total_preds, preds[:, :len(label_names)]))

    torch.cuda.empty_cache()
    return total_preds

def save_results2csv(label_names, precs, recs, f1s, aucs, note_type):
    save_data = {
            'label': label_names,
            'valid_precision': precs.tolist()[0],
            'valid_recall': recs.tolist()[0],
            'valid_f1': f1s.tolist()[0],
            'valid_auc': aucs.tolist()[0],
        }
    save_df = pl.DataFrame(save_data)
    try: # assume we are running with srun/sbatch with a retrieveable job id
        job_id = os.environ['SLURM_JOB_ID']
        save_df.write_csv('results/result_{}_{}.csv'.format(note_type, job_id))
    except:
        save_df.write_csv('results/result_{}.csv'.format(note_type))

def plot_summary_writer(epochs, train_losses, test_losses, label_names, note_type, test_precs, test_recs, test_f1s, test_aucs):
    writer = SummaryWriter()

    for epoch in range(epochs):
        writer.add_scalar('Loss/Train/', train_losses[epoch], epoch)
        writer.add_scalar('Loss/testation/', test_losses[epoch], epoch)
        for label_num, label_name in enumerate(label_names):
            writer.add_scalar('Precision/Test/{}/{}/'.format(note_type, label_name), test_precs[epoch, label_num], epoch)
            writer.add_scalar('Recall/Test/{}/{}/'.format(note_type, label_name), test_recs[epoch, label_num], epoch)
            writer.add_scalar('F1/Test/{}/{}/'.format(note_type, label_name), test_f1s[epoch, label_num], epoch)
            writer.add_scalar('AUC/Test/{}/{}/'.format(note_type, label_name), test_aucs[epoch, label_num], epoch)

def calculate_f1_prec_rec_auc(label_names: list, total_preds, total_labels, epoch, best_model_eval_weight_path, save_csv=False, avg='micro', auc='roc'):
    assert auc == 'roc' or auc == 'pr', "InputError: 'auc' needs to be either 'roc' or 'pr'!"

    auc_scores, total_preds_binary, f1_scores, precisions, recalls = [], [], [], [], []
    for i in range(len(label_names)):
        preds_i, labels_i = total_preds[:, i], total_labels[:, i]
        preds_i = preds_i.cpu() if preds_i.is_cuda else preds_i
        labels_i = labels_i.cpu() if labels_i.is_cuda else labels_i
        auc_scores_i = roc_auc_score(labels_i, preds_i) if auc == 'roc' else average_precision_score(labels_i, preds_i)
        auc_scores.append(auc_scores_i)
        precision_i, recall_i, thresholds = precision_recall_curve(y_true=labels_i, probas_pred=preds_i)
        optimal_thresholds, preds_binary_i = optimal_threshold_precision_recall_curve(preds=preds_i, precision=precision_i, recall=recall_i, thresholds=thresholds)
        # opt_precision_i, opt_recall_i, opt_f1_scores_i = compute_precision_recall_f1(predictions=torch.tensor(preds_binary_i), targets=labels_i)                
        precision_i, recall_i, f1_score_i = precision_score(preds_binary_i, labels_i), recall_score(preds_binary_i, labels_i), f1_score(preds_binary_i, labels_i)
        f1_scores.append(f1_score_i)
        precisions.append(precision_i)
        recalls.append(recall_i)
        total_preds_binary = torch.tensor(preds_binary_i) if i == 0 else torch.vstack((total_preds_binary, torch.tensor(preds_binary_i)))

    if save_csv:
        labels_df, preds_df = pd.DataFrame(total_labels), pd.DataFrame(np.transpose(total_preds_binary.numpy()))
        save_batch_path = os.path.abspath(os.path.join(best_model_eval_weight_path, os.pardir))
        labels_df.to_csv(save_batch_path+'/'+'batch_labels_epoch{}.csv'.format(epoch))
        preds_df.to_csv(save_batch_path+'/'+'batch_preds_epoch{}.csv'.format(epoch))

    auc_avg = roc_auc_score(y_true=total_labels, y_score=total_preds, average=avg) if auc == 'roc' else average_precision_score(y_true=total_labels, y_score=total_preds, average=avg)
    f1_avg = f1_score(y_true=total_labels, y_pred=torch.transpose(total_preds_binary, 0, 1), average=avg)
    precision_avg, recall_avg, _, _ = precision_recall_fscore_support(y_true=total_labels, y_pred=torch.transpose(total_preds_binary, 0, 1), average=avg)

    return auc_scores, f1_scores, precisions, recalls, auc_avg, f1_avg, precision_avg, recall_avg

def find_closest_mixmax_power2(min: int, max: int):
    '''
    find the minimum and maximum value that is a power of 2
    '''
    assert min * 2 <= max, 'InputError: max should be at least 2 times larger than min'
    min_exp = int(np.ceil(np.log2(min)))
    max_exp = int(np.floor(np.log2(max)))
    res_power2s = [2**ele for ele in list(range(min_exp, max_exp+1))]
    
    return res_power2s

def filter_col_with_substring(df: pl.DataFrame, col_name: str, pattern: str, case_sensitive=True):
    '''
    filter a column that contains a certain substring
    Args:
        df (pl.DataFrame): a dataframe to fiter
        col_name (str): a column name that lives in the dataframe
        pattern (str): a regular expression pattern in string format
        case_sensitive (bool): whether or not the regular expression pattern is case sensitive
    Returns: 
        pl.DataFrame with filtered column that contains the desired substring pattern
    '''
    case_sensitive_str = '' if case_sensitive else '(?i)'

    return df.filter(
        pl.col(col_name).str.contains(case_sensitive_str+pattern)
    )

def find_longest_common_substring(text1, text2):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).find_longest_match(0, len(text1), 0, len(text2))
