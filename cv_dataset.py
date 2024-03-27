from transformers import LlamaTokenizer, BertTokenizer
from utils import (
    tokenize, count_parameters, compute_metrics,
    optimal_threshold_precision_recall_curve, compute_precision_recall_f1,
    initialize_medspacy_nlp_pipeline, get_procedure_section, remove_nondiagnostic_info,
    remove_punctuation, sentence_splitter, flatten_tokens,
    report_gpu, get_latest_checkpoint_file, calculate_ci,
    visualize_shap, get_false_predictions, infer_from_pretrained,
    save_results2csv, plot_summary_writer, get_medical_history_sections
)
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import SequentialSampler
from sklearn.model_selection import train_test_split
from pprint import pprint
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df: pl.DataFrame, tokenizer, divide_sent):
        self.df = df
        self.nlp = initialize_medspacy_nlp_pipeline()
        self.tokenizer = tokenizer
        self.divide_sent = divide_sent
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        assert isinstance(idx, int) and (idx >= 0 and idx < len(self.df)), f'idx must be an integer in between 0 and len(self.df)-1! current idx : {idx}, length: {len(self.df)}'

        # use __call__ method of the tokenizer that generates a dictionary with input_ids, token_type_ids, and attention_mask
        if self.divide_sent: # divide into sentences and flatten out the tokens
            _text = remove_punctuation(sentence_splitter(report=self.df[idx, 'CONCAT_NOTE_TEXT'], min_len=3, nlp=self.nlp))
            tokens = self.tokenizer( 
                text=_text, max_length=512, padding=False, truncation=True, return_token_type_ids=False
            )
            tokens = flatten_tokens(token=tokens, max_length=512)
            return {
                'input_ids' : torch.tensor(tokens['input_ids'], dtype=torch.int),
                'attention_mask' : torch.tensor(tokens['attention_mask'], dtype=torch.int),
                'labels' : torch.tensor(list(self.df.drop('CONCAT_NOTE_TEXT').row(idx)), dtype=torch.int)         
            }
        else:
            tokens = self.tokenizer( 
                text=self.df[idx, 'CONCAT_NOTE_TEXT'], max_length=512, padding='max_length', truncation=True, return_token_type_ids=False
            ) 
            return {
                'input_ids' : torch.tensor(tokens['input_ids'], dtype=torch.int),
                'attention_mask' : torch.tensor(tokens['attention_mask'], dtype=torch.int),
                'labels' : torch.tensor(list(self.df.drop('CONCAT_NOTE_TEXT').row(idx)), dtype=torch.int)         
            }

class CVDataset(torch.utils.data.Dataset):
    '''
    torch.utils.data.Dataset is an abstract class representing a dataset.
    Custom dataset should inherit Dataset and override the following methods:
        __len__ so that len(dataset) returns the size of the dataset.
        __getitem__ to support the indexing such that dataset[i] can be used to get ith sample.
    Read the csv in __init__ but leave the reading text to __getitem__. 
    This is memory efficient since the texts are not stored in the memory at once but read as required.
    '''
    def __init__(
            self, label_names, note_type: list, model_type: str, random_split: bool, raw_registry: bool, 
            validate: bool, llama_model_name: str, sectionize: list, bert_model_name='bert-base-uncased', divide_sent=True, 
            save_batch_path='', preprocess=True, predict_on='surgery', concatenate=True
        ):
        assert isinstance(label_names, list) or isinstance(label_names, str), "argument 'label_names' must be either type string or list!"
        assert model_type == 'bert' or model_type == 'llama', "ValueError: 'model_type' must be either 'bert' or 'llama'!"
        self.label_names = label_names
        self.note_type = note_type
        self.model_type = model_type
        self.divide_sent = divide_sent
        self.random_split = random_split
        self.raw_registry = raw_registry
        self.save_batch_path = save_batch_path
        self.validate = validate
        self.preprocess = preprocess
        self.predict_on = predict_on
        self.nlp = initialize_medspacy_nlp_pipeline()
        self.identifier_lis = ['Surg_Dt', 'PAT_MRN_ID']
        self.sectionize = sectionize
        self.concatenate = concatenate


        if 'operative_report' in self.note_type: # only 'OPERATIVE REPORT'
            self.join_df  = pl.read_csv('/data/aiiih/projects/avr/data/cv_cohort_op_notes.csv').drop_nulls('NOTE_TEXT').with_columns(
                IP_NOTE_DESC = pl.lit('OPERATIVE REPORT')
            )
            # self.label_names = [col for col in self.join_df.columns if 'SP_' in col]
        elif 'op_notes' in self.note_type: # multiple op note types
            self.join_df  = pl.read_csv('/data/aiiih/projects/avr/data/cv_cohort_full_notes.csv').filter(
                (pl.col('IP_NOTE_DESC') == 'OPERATIVE REPORT') | (pl.col('IP_NOTE_DESC') == 'BRIEF OP NOTE') | (pl.col('IP_NOTE_DESC') == 'PROCEDURES')
            )
        if 'h&p' in self.note_type: 
            try:
                self.join_df
            except: # self.join_df wasn't defined
                self.join_df = pl.read_csv('/data/aiiih/projects/jl/data/comorbidity/cv_cohort_hnp_comorbidity_notes.csv').filter(
                    (pl.col('IP_NOTE_DESC') == 'H&P') | (pl.col('IP_NOTE_DESC') == 'H&P (VIEW-ONLY)')
                )
            else: # self.join_df was defined
                self.join_df = self.join_df.vstack(
                    pl.read_csv('/data/aiiih/projects/jl/data/comorbidity/cv_cohort_hnp_comorbidity_notes.csv').filter(
                        (pl.col('IP_NOTE_DESC') == 'H&P') | (pl.col('IP_NOTE_DESC') == 'H&P (VIEW-ONLY)')
                    )
                )
        if 'consults' in self.note_type: 
            try:
                self.join_df
            except: # self.join_df wasn't defined
                self.join_df = pl.read_csv('/data/aiiih/projects/jl/data/comorbidity/cv_cohort_consults_comorbidity_notes.csv').filter(
                    pl.col('IP_NOTE_DESC') == 'CONSULTS'
                )
            else: # self.join_df was defined
                self.join_df = self.join_df.vstack(
                    pl.read_csv('/data/aiiih/projects/jl/data/comorbidity/cv_cohort_consults_comorbidity_notes.csv').filter(
                        pl.col('IP_NOTE_DESC') == 'CONSULTS'
                    )
                )
        if 'discharge' in self.note_type: 
            try:
                self.join_df
            except: # self.join_df wasn't defined
                self.join_df = pl.read_csv('/data/aiiih/projects/jl/data/comorbidity/cv_cohort_discharge_summary_comorbidity_notes.csv').filter(
                    pl.col('IP_NOTE_DESC') == 'DISCHARGE SUMMARY'
                )
            else: # self.join_df was defined
                self.join_df = self.join_df.vstack(
                    pl.read_csv('/data/aiiih/projects/jl/data/comorbidity/cv_cohort_discharge_summary_comorbidity_notes.csv').filter(
                        pl.col('IP_NOTE_DESC') == 'DISCHARGE SUMMARY'
                    )
                )
            
        if self.predict_on == 'surgery':
            full_sp_df = pl.read_csv('/data/aiiih/projects/jl/cardsurg/av_pilot_dl/full_sp_vars.csv')
            full_sp_df = full_sp_df.select( # select columns if they are requested in label_names
                list(set(self.label_names) - set(self.join_df.columns)) + ['PAT_MRN_ID', 'Surg_Dt']
            )
            self.join_df = self.join_df.join(full_sp_df, on=['Surg_Dt', 'PAT_MRN_ID'], how='inner').select(
                self.label_names + [col for col in self.join_df.columns if 'SP_' not in col]
            )
        elif self.predict_on == 'comorbidity':
            csc_df = pl.read_csv('/data/aiiih/projects/jl/data/comorbidity/cardsurg_comorbidities.csv')
            csc_df = csc_df.drop([col for col in csc_df.columns if 'PREOP_' in col or 'SP_' in col]) 
            self.join_df = self.join_df.join(csc_df, on=['PAT_MRN_ID', 'Surg_Dt'], how='inner').drop([col for col in self.join_df.columns if 'SP_' in col])

        if self.concatenate:
            # Concatentate notes by aggregating NOTE_TEXT with different LINEs
            self.join_df = self.join_df.drop_nulls('NOTE_TEXT').unique(subset=['LINE', 'IP_NOTE_DESC', 'K_NOTE_KEY', 'PAT_MRN_ID'])
            self.join_df = self.join_df.sort(['LINE', 'IP_NOTE_DESC', 'K_NOTE_KEY', 'PAT_MRN_ID'])
            self.join_df = self.join_df.to_pandas()
            self.join_df['CONCAT_NOTE_TEXT'] = self.join_df.groupby(['K_NOTE_KEY', 'IP_NOTE_DESC', 'PAT_MRN_ID'])['NOTE_TEXT'].transform(lambda x : '\n'.join(x))
            self.join_df = pl.from_pandas(self.join_df).unique(subset=['K_NOTE_KEY', 'IP_NOTE_DESC'])#.select([col for col in self.join_df.columns if 'SP_' in col or col == 'CONCAT_NOTE_TEXT'])

        else: # if not concatenate, we just retrieve the first 'LINE' from 'NOTE_TEXT'
            self.join_df = self.join_df.drop_nulls('NOTE_TEXT').filter(pl.col('LINE')==1).rename({'NOTE_TEXT': 'CONCAT_NOTE_TEXT'})
            self.join_df = self.join_df.unique(subset=['PAT_MRN_ID', 'Surg_Dt'])
            

        if self.raw_registry:
            self.registry_df = pl.read_csv('/data/aiiih/projects/avr/data/sts_420_procs.csv')
            self.join_df = self.join_df.join(self.registry_df, on=['Surg_Dt', 'K_PAT_KEY'], how='inner')
            self.join_df = self.join_df.unique(subset=['PAT_MRN_ID', 'Surg_Dt'])

            join_df = self.join_df.to_pandas()
            for label in label_names: # replace with "NULL" in column values that occur less than 1% within a column
                relative_frequencies = join_df[label].value_counts(normalize=True)
                # filtered_vals = relative_frequencies[relative_frequencies >= 0.01].index.tolist()
                unfiltered_vals = relative_frequencies[relative_frequencies < 0.03].index.tolist()
                null_lis = [join_df[join_df[label]==uv].index.values.tolist() for uv in unfiltered_vals]
                null_lis = [item for sublist in null_lis for item in sublist]
                join_df.loc[null_lis, label] = 'NULL'
                # join_df = join_df[join_df[label].apply(lambda x: x in filtered_vals)]
            self.join_df = pl.from_pandas(join_df)
            self.label_lis = []
            for label in label_names:
                self.join_df = self.join_df.with_columns(
                    [
                        pl.when(pl.col(label) == val)
                        .then(1)
                        .otherwise(0)
                        .alias('{}_{}'.format(label, val))
                        for val in self.join_df[label].unique().to_list()
                    ]
                )
                self.label_lis.extend(['{}_{}'.format(label, val) for val in self.join_df[label].unique().to_list()])
                
            self.label_names = self.label_lis
        else: 
            self.label_lis = [col for col in self.join_df.columns if any(prefix in col for prefix in ['SP_', 'HX_', 'CB_', 'Preop_'])]

        # type-casting the labels to binary variables
        self.join_df = self.join_df.with_columns(
            [
                pl.col(label_col).cast(pl.Int32).keep_name() for label_col in self.label_lis
            ]
        )

        self.join_df = self.join_df.fill_null(0).select(self.identifier_lis + self.label_lis + ['CONCAT_NOTE_TEXT'])#.sample(frac=0.05)

        if len(self.sectionize) > 0 and isinstance(self.sectionize, list):
            if 'medical history' in self.sectionize:
                self.join_df = self.join_df.with_columns([
                    pl.col('CONCAT_NOTE_TEXT').apply(lambda s: get_medical_history_sections(report=s)).alias('CONCAT_NOTE_TEXT')
                ])
            
            if 'procedure' in self.sectionize:
                self.join_df = self.join_df.with_columns([
                    pl.col('CONCAT_NOTE_TEXT').apply(lambda s: get_procedure_section(report=s)).alias('CONCAT_NOTE_TEXT')
                ])

        if self.preprocess:
            ################## TEXT DATA PREPROCESSING ###################
            self.join_df = self.join_df.with_columns([
                pl.col('CONCAT_NOTE_TEXT').apply(lambda s: remove_nondiagnostic_info(report=s, _remove_names=False)).alias('CONCAT_NOTE_TEXT')
            ]).drop_nulls('CONCAT_NOTE_TEXT')

        # drop empty and null strings (first convert empty strings to null and drop nulls)
        self.join_df = self.join_df.with_columns(
            [
                pl.when(pl.col('CONCAT_NOTE_TEXT').str.lengths()==0)
                .then(None)
                .otherwise(pl.col('CONCAT_NOTE_TEXT'))
                .keep_name()
            ] 
        ).drop_nulls('CONCAT_NOTE_TEXT')

        if not self.random_split: # merge with sts version dataframe when train/test split is based on external sts variables
            sts_version_df = pl.read_csv('/data/aiiih/projects/avr/data/sts_card_procs.csv', ignore_errors=True).select(
                ['PAT_MRN_ID', 'Surg_Dt', 'Hospital', 'STSVersion']
            )
            self.join_df = self.join_df.join(sts_version_df, on=self.identifier_lis, how='inner').with_row_count(name='index')
        # Initialize Tokenizer
        if self.model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name) # default tokenizer from pretrained BERT models
            # new_tokenizer = Tokenizer.from_file('models/card_tok_half_10/tokenizer.json') 
            # self.tokenizer = BertTokenizerFast(tokenizer_object=new_tokenizer)
        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token # (0)

        print('INITIALIED A CV DATASET!', flush=True)

    def __len__(self):
        return len(self.join_df)

    def __getitem__(self, idx: int):
        assert isinstance(idx, int) and (idx >= 0 and idx < len(self.join_df)), 'idx must be an integer in between 0 and len(self.join_df)-1!'

        # use __call__ method of the tokenizer that generates a dictionary with input_ids, token_type_ids, and attention_mask
        if self.divide_sent: # divide into sentences and flatten out the tokens
            _text = remove_punctuation(sentence_splitter(report=self.join_df[idx, 'CONCAT_NOTE_TEXT'], min_len=3, nlp=self.nlp))
            tokens = self.tokenizer( 
                text=_text, max_length=512, padding=False, truncation=True, return_token_type_ids=False
            )
            tokens = flatten_tokens(token=tokens, max_length=512)
            return {
                'input_ids' : torch.tensor(tokens['input_ids'], dtype=torch.int),
                'attention_mask' : torch.tensor(tokens['attention_mask'], dtype=torch.int),
                'labels' : torch.tensor(list(self.join_df.drop('CONCAT_NOTE_TEXT').row(idx)), dtype=torch.int)         
            }
        else:
            tokens = self.tokenizer( 
                text=self.join_df[idx, 'CONCAT_NOTE_TEXT'], max_length=512, padding='max_length', truncation=True, return_token_type_ids=False
            ) 
            return {
                'input_ids' : torch.tensor(tokens['input_ids'], dtype=torch.int),
                'attention_mask' : torch.tensor(tokens['attention_mask'], dtype=torch.int),
                'labels' : torch.tensor(list(self.join_df.drop('CONCAT_NOTE_TEXT').row(idx)), dtype=torch.int)         
            }

    def my_train_test_split(self, shuffle):
        if self.random_split: # mix train and test sets randomly 
            train_test_split_ratio, dataset_size = 0.8, len(self.join_df) 
            indices = list(range(dataset_size))
            if self.validate:
                test_val_split_ratio = ((1.0-train_test_split_ratio)/2 + train_test_split_ratio) # (train, validation, test): (0.8, 0.1, 0.1)
                train_split, val_test_split = int(np.floor(train_test_split_ratio * dataset_size)), int(np.floor(test_val_split_ratio * dataset_size)) 
            else:
                train_split = int(np.floor(train_test_split_ratio * dataset_size))

            if shuffle:
                np.random.seed(42)
                np.random.shuffle(indices)

            # reorder columns for later train_val_loop in TextClassifier
            self.join_df = self.join_df.select(self.label_lis + ['CONCAT_NOTE_TEXT'])

            if self.validate:
                train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_test_split], indices[val_test_split:]  # Creating data indices for training and validation splits
                train_df, valid_df, test_df = self.join_df[train_indices, :], self.join_df[valid_indices, :], self.join_df[test_indices, :]
                self.train_dataset, self.valid_dataset = CustomDataset(train_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent), CustomDataset(valid_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent) 
                self.test_dataset = CustomDataset(test_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent)
                return SequentialSampler(train_indices), SequentialSampler(val_indices), SequentialSampler(test_indices), train_indices, val_indices, test_indices # create and return PT train, validation, and teste samplers
            else:
                train_indices, test_indices = indices[:train_split], indices[train_split:]
                train_df, test_df = self.join_df[train_indices, :], self.join_df[test_indices, :]
                self.train_dataset, self.test_dataset = CustomDataset(train_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent), CustomDataset(test_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent)
                return SequentialSampler(train_indices), SequentialSampler(test_indices), train_indices, test_indices
        
        else:

            train_indices = self.join_df.filter((pl.col('Hospital') == 'Cleveland Clinic') & (pl.col('STSVersion') != 4.2)).select('index').to_series().to_list() 
            test_indices = self.join_df.filter((pl.col('Hospital') == 'Cleveland Clinic') & (pl.col('STSVersion') == 4.2)).select('index').to_series().to_list() 
            if shuffle:
                random.shuffle(train_indices)
                random.shuffle(test_indices)

            # reorder columns for later train_val_loop in TextClassifier
            self.join_df = self.join_df.select(self.label_names + ['CONCAT_NOTE_TEXT'])
            if self.validate:
                train_indices, valid_indices = train_indices[round(len(train_indices) * 0.15):], train_indices[:round(len(train_indices) * 0.15)]
                train_df, valid_df, test_df = self.join_df[train_indices, :], self.join_df[valid_indices, :], self.join_df[test_indices, :]
                self.train_dataset, self.valid_dataset = CustomDataset(train_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent), CustomDataset(valid_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent) 
                self.test_dataset = CustomDataset(test_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent)

                if self.save_batch_path != '':
                    test_df.write_csv(self.save_batch_path+'/'+'test_data.csv')

                return SequentialSampler(train_df), SequentialSampler(valid_df), SequentialSampler(test_df), train_indices, valid_indices, test_indices
            else:
                train_df, test_df = self.join_df[train_indices, :], self.join_df[test_indices, :]
                self.train_dataset, self.test_dataset = CustomDataset(train_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent), CustomDataset(test_df, tokenizer=self.tokenizer, divide_sent=self.divide_sent)

                if self.save_batch_path != '':
                    test_df.write_csv(self.save_batch_path+'/'+'test_data.csv')

                return SequentialSampler(train_df), SequentialSampler(test_df), train_indices, test_indices
