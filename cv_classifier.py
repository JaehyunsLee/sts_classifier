from transformers import ( 
    BertModel, LlamaTokenizer, LlamaForSequenceClassification,
    BertTokenizer, AutoModelForCausalLM, Trainer, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments, IntervalStrategy,
    EarlyStoppingCallback, DataCollatorWithPadding
)
from peft import (
    prepare_model_for_int8_training, LoraConfig, get_peft_model, 
    get_peft_model_state_dict
)
from utils import (
    tokenize, count_parameters, compute_metrics,
    optimal_threshold_precision_recall_curve, compute_precision_recall_f1,
    initialize_medspacy_nlp_pipeline, get_procedure_section, remove_nondiagnostic_info,
    remove_punctuation, sentence_splitter, flatten_tokens,
    report_gpu, get_latest_checkpoint_file, calculate_ci,
    visualize_shap, get_false_predictions, infer_from_pretrained,
    save_results2csv, plot_summary_writer, calculate_f1_prec_rec_auc,
    find_closest_mixmax_power2
)
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_recall_curve,
    precision_recall_fscore_support, precision_score,
    recall_score
)
from torch.nn import (
    Dropout, Linear, Sigmoid, Softmax, 
    GELU, ReLU, SELU, ELU, RReLU, LeakyReLU, 
    BCELoss, CrossEntropyLoss
)
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.ensemble import RandomForestClassifier
from torch.optim import AdamW
from pathlib import Path
from cv_dataset import CVDataset
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import polars as pl
import pandas as pd
import numpy as np
import datetime
import torch
import sys
import gc
import os

class RandomForestSTSClassifier():
    def __init__(self, label_names: list, data: pl.DataFrame, train_idx: list, val_idx: list, average: str):
        cpt_df = pd.read_csv('/data/aiiih/projects/avr/data/avr_cohort_cpt.csv')
        self.label_names, x_col = label_names, [col for col in cpt_df.columns if 'CPT_' in col]
        cpt_df['Surg_Dt'] = pd.to_datetime(cpt_df.Surg_Dt)
        cpt_df = pl.from_pandas(cpt_df).with_columns(
            pl.col('Surg_Dt').cast(pl.Utf8).str.replace(" 00:00:00.000000000", "")
        ).rename({'CCF_MRN': 'PAT_MRN_ID'}).with_columns(
            pl.col(col_name).fill_null(value=0)
            for col_name in x_col
        )
        
        train, test = data[train_idx, :].drop('CONCAT_NOTE_TEXT'), data[val_idx, :].drop('CONCAT_NOTE_TEXT')
        train, test = train.join(cpt_df, on=['Surg_Dt', 'PAT_MRN_ID'], how='inner'), test.join(cpt_df, on=['Surg_Dt', 'PAT_MRN_ID'], how='inner')
        X_train, X_test = train.select(x_col).with_columns(pl.when(pl.col(pl.Utf8) == None).then(0).otherwise(pl.col(pl.Utf8)).keep_name()), test.select(x_col).with_columns(pl.when(pl.col(pl.Utf8) == None).then(0).otherwise(pl.col(pl.Utf8)).keep_name())
        y_train, y_test = train.select(self.label_names), test.select(self.label_names)
        X_train, X_test = X_train.with_columns(pl.col(col).cast(pl.Int64, strict=False) for col in X_train.columns), X_test.with_columns(pl.col(col).cast(pl.Int64, strict=False) for col in X_test.columns)
        X_train = X_train.with_columns(
            pl.col(col_name).fill_null(value=0)
            for col_name in x_col
        )
        X_test = X_test.with_columns(
            pl.col(col_name).fill_null(value=0)
            for col_name in x_col
        )

        rf = RandomForestClassifier()
        rf.n_outputs_= len(self.label_names)
        rf.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred, labels = rf.predict(X_test.to_numpy()), y_test.to_numpy()
        y_proba = np.array(rf.predict_proba(X_test.to_numpy()))
        y_proba = np.transpose([pred[:, 1] for pred in y_proba])
        avg_prec, prec = precision_score(y_pred, labels, average=average), [precision_score(y_pred[:, i], labels[:, i]) for i in range(labels.shape[1])]
        avg_rec, rec = recall_score(y_pred, labels, average=average), [recall_score(y_pred[:, i], labels[:, i]) for i in range(labels.shape[1])]
        avg_f1, f1 = f1_score(y_pred, labels, average=average), [f1_score(y_pred[:, i], labels[:, i]) for i in range(labels.shape[1])]
        avg_auc, auc = roc_auc_score(labels, y_proba, average=average), [roc_auc_score(labels[:, i], y_proba[:, i]) for i in range(labels.shape[1])]

        print(f'RANDOM FOREST RESULTS: \nLabels: {self.label_names} \nAUC: {auc} \nF1: {f1} \nPRECISION: {prec} \nRECALL: {rec} \nmicro-AUC: {avg_auc} \n{average}-F1: {avg_f1} \n{average}-PRECISION: {avg_prec} \n{average}-RECALL: {avg_rec}')


class LlamaClassifier():

    def __init__(self, label_names: list, llama_model_name="meta-llama/Llama-2-7b-chat-hf", note_type='op_notes', divide_sent=True, random_split=False, raw_registry=False, validate=False, **kwargs):
        super(LlamaClassifier, self).__init__()
        assert validate == True, 'validate has to be True!'
        self.cv_dataset = CVDataset(
            label_names=label_names, bert_model_name='bert-base-uncased', llama_model_name=llama_model_name, 
            note_type=note_type, model_type='llama', divide_sent=divide_sent, random_split=random_split, validate=validate,
            raw_registry=raw_registry
        )

        join_df = self.cv_dataset.join_df.select(['CONCAT_NOTE_TEXT']+ label_names)
        self.train_sampler, self.valid_sampler, self.test_sampler, self.train_indices, self.valid_indices, self.test_indices = self.cv_dataset.my_train_test_split(shuffle=True)
        # val_idx, train_idx, test_idx = self.train_sampler.indices[:round(len(self.train_sampler.indices)*0.25)], self.train_sampler.indices[round(len(self.train_sampler.indices)*0.25):], self.test_sampler.indices # 0.6, 0.2, 0.2
        train_dataset, val_dataset, test_dataset = join_df[self.train_indices, :].to_pandas(), join_df[self.valid_indices, :].to_pandas(), join_df[self.test_indices, :].to_pandas()

        self.dataset = DatasetDict({
            'train': Dataset.from_dict(train_dataset),
            'validation': Dataset.from_dict(val_dataset),
            'test' : Dataset.from_dict(test_dataset),
        })
        self.label_names = label_names
        self.id2label = {idx:label for idx, label in enumerate(self.label_names)}
        self.label2id = {v: k for k, v in self.id2label.items()} 
        self.output_dir = 'models/llama_classifier/'
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token # (0)
        self.encoded_dataset = self.dataset.map(tokenize, batched=True, remove_columns=self.label_names + ['CONCAT_NOTE_TEXT'])
        self.encoded_dataset.set_format('torch')

        example = self.encoded_dataset['train'][0]

        # INITIALIZE HUGGINGFACE QUANTIZATION and LORA CONFIGURATIONS HERE!
        nf4_config  = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True, # nested quantization (saves an additional 0.4 bits per parameter)
            bnb_4bit_quant_type='nf4', # normalized 4bit datatype adapted for weights that have been initilized using a normal distribution (can use pure "FP4" quantization)
            bnb_4bit_compute_dtype=torch.bfloat16 # while 4-bit bitsandbytes stores weights in 4-bits, the computation still happens in 16-bits
        )
        self.model = LlamaForSequenceClassification.from_pretrained(
            llama_model_name, device_map='auto',
            quantization_config=nf4_config, problem_type='multi_label_classification', # makes sure the appropriate loss function is used (BCEWithLogitsLoss)
            num_labels=len(self.label_names),
            id2label=self.id2label, label2id=self.label2id
        )
        count_parameters(self.model)

        self.model = prepare_model_for_int8_training(self.model) # prepare model for training wtih LoRA algorithm
        self.lora_config = LoraConfig(
            r=64, # rank of the update matrices; lora attention dimension
            target_modules=["q_proj","v_proj"], # modules (ex. attention blocks) to apply the LoRA update matrices.
            lora_alpha=16, # Lora scaling: regularization strength
            lora_dropout=0.05, # dropout probability of Lora layers. 
            bias='none', # Bias type for Lora. Can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training.
            task_type='SEQ_CLS'
        )
        self.model = get_peft_model(self.model, self.lora_config)
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2
        self.model.config.pad_token_id = self.tokenizer.pad_token_id = self.model.config.eos_token_id
        count_parameters(self.model)
        self.model.print_trainable_parameters()
        print('INITIALIZED A LLaMA CLASSIFIER!')

    def train_eval_llama_classifier(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            # label_names=self.label_names,
            evaluation_strategy=IntervalStrategy.STEPS,   # "steps"
            save_strategy='steps',
            overwrite_output_dir=True,      # Use this to continue training if output_dir points to a checkpoint directory.
            remove_unused_columns=False,    # Whether or not to automatically remove the columns unused by the model forward method.
            num_train_epochs=10,            # number of training epochs, feel free to tweak
            per_device_train_batch_size=256, # The batch size per GPU/TPU core/CPU for training.
            per_device_eval_batch_size=128,  # The batch size per GPU/TPU core/CPU for evaluation.
            # gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights. Instead of calculating the gradients for the whole batch at once to do it in smaller steps
            # gradient_checkpointing=True,    # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
            auto_find_batch_size=True,      # Whether to find a batch size that will fit into memory automatically through exponential decay, avoiding CUDA Out-of-Memory errors.
            logging_steps=10,             # evaluate, log and save model checkpoints every 1000 step
            logging_strategy='steps',       # Logging is done at the end of each epoch.
            save_steps=500,
            warmup_steps=100,             
            learning_rate=2e-4,
            lr_scheduler_type='cosine',
            warmup_ratio=0.03,              
            weight_decay=0.001,
            optim='adamw_torch',       
            load_best_model_at_end=True,    # whether to load the best model (in terms of loss) at the end of training
            save_total_limit=3,             # whether you don't have much space so you let only 1 model weights saved in the disk
            bf16=False,                      # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
            fp16=True,
            bf16_full_eval=False,            #  Whether to use full bfloat16 evaluation instead of 32-bit. This will be faster and save memory but can harm metric values. This is an experimental API and it may change.
            fp16_full_eval=True,
            report_to='tensorboard',      # The list of integrations to report the results and logs to.
            seed=42,
            disable_tqdm=False
        )

        data_collator = DataCollatorWithPadding( # Data collators are objects that will form a batch by using a list of dataset elements as input
            tokenizer=self.tokenizer, padding='max_length', max_length=512, return_tensors='pt' # pad_to_multiple_of=512
        )

        output_ex = self.model(input_ids=self.encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=self.encoded_dataset['train']['labels'][0].unsqueeze(0)) # forward pass

        self.trainer = Trainer(
            model=self.model,
            train_dataset=self.encoded_dataset['train'],
            eval_dataset=self.encoded_dataset['validation'],
            data_collator=data_collator,
            # peft_config=self.lora_config,
            # dataset_text_field="CONCAT_NOTE_TEXT",
            tokenizer=self.tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            callbacks= [EarlyStoppingCallback(early_stopping_patience=3)]
        )
        self.model.config.use_cache = False
        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(self.model, type(self.model))

        # self.model = torch.compile(self.model) # compiles the model's computation graph and prepares it for training using torch

        print('Training...')
        self.trainer.train()
        self.trainer.save_model(output_dir=self.output_dir)
        # self.model.save_pretrained(self.output_dir)
        
        print('Evaluating...')
        self.trainer.evaluate() # predict + compute metrics on test set
        # self.trainer.predict(self.dataset['validation']) # only predict labels on test set (also compute metrics if test set contains ground-truth labels)

        # self.model = self.model.eval()
        # self.model = torch.compile(self.model)

        self.trainer.save_model(output_dir=self.output_dir)
        # self.model.save_pretrained(self.output_dir)

    def infer_llama_classifier(self, text: str):
        encoding = self.tokenizer(text, return_tensors='pt')
        encoding = {k: v.to(self.trainer.model.device) for k,v in encoding.items()}

        outputs = self.trainer.model(**encoding) 
        ''' 
        The logits that come out of the model are of shape (batch_size, num_labels).
        As we are only forwarding a single sentence through the model, the batch_size equals 1. 
        The logits is a tensor that contains the (unnormalized) scores for every individual label.
        '''
        logits = outputs.logits
        print('LOGITS SHAPE: ', logits.shape)

        # apply sigmoid + threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        # turn predicted id's into actual label names
        predicted_labels = [self.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
        print('PREDICTED LABLES: ', predicted_labels)


class BertClassifier(torch.nn.Module):
    def __init__(self, n_labels: int, multi_class=1, dropout=0.1, l1=512, bert_model_name='bert-base-uncased', activation_fn='leaky_relu', **kwargs):
        super(BertClassifier, self).__init__()
        self.multi_class = multi_class
        self.validate, self.l1 = False, l1

        # bert models ex: 'bert-base-uncased', 'dmis-lab/biobert-base-cased-v1.2', "dmis-lab/biobert-v1.1", "emilyalsentzer/Bio_ClinicalBERT"
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_layer_num = sum(1 for _ in self.bert.named_parameters())

        # freeze certain number (%) of bert layers
        self.num_unfreeze_layer, self.ratio_unfreeze_layer = self.bert_layer_num, 0.0
        if kwargs: # check if a variable was passed as kwargs 
            for key, value in kwargs.items():
                if key == 'unfreeze' and isinstance(value, float): # percentage of unfrozen bert layers
                    assert value >= 0.0 and value <= 1.0, 'ValueError: value must be a ratio between 0.0 and 1.0'
                    self.ratio_unfreeze_layer = value
                if key == 'l1': # whether or not validation mode is on
                    print(key, value)
                    assert value % 2 == 0 and (value < 768 and value >= n_labels), 'ValueError: value must be in between [2, 767]'
                    self.l1, self.validate = value, True

        if self.ratio_unfreeze_layer > 0.0:
            self.num_unfreeze_layer = int(self.bert_layer_num * self.ratio_unfreeze_layer)
            # freeze the pretrained part of BertModel.from_pretrained model
            for param in list(self.bert.parameters())[:-self.num_unfreeze_layer]: 
                param.requires_grad = False
        else:
            for param in list(self.bert.parameters()): 
                param.requires_grad = False

        self.dropout = Dropout(dropout)
        self.fc1 = Linear(768, self.l1) # dense layer 1: 768 comes from the BERT hidden size (bert_model.config.hidden_size = 768)
        self.fc2 = Linear(self.l1, self.multi_class) if self.multi_class > 2 else Linear(self.l1, n_labels) # dense layer 2 (output layer)
        self.sigmoid = Sigmoid() # only for binary classification, softmax is for multiclass classifiers
        self.softmax = Softmax() # only for multiclass classification; each neuron in the output layer yields a probability for the corresponding class
        self.activation_fn_dict = {
            'leaky_relu': LeakyReLU(), 'gelu': GELU(), 'relu': ReLU(), 
            'selu': SELU(), 'elu': ELU(), 'rrelu': RReLU()
        }
        self.activation_fn = self.activation_fn_dict[activation_fn] 

        count_parameters(self.bert)
        print('INITIALIZED A BERTCLASSIFIER!')

    def forward(self, sent_id, mask): # define the forward pass
        _, pooled_output = self.bert(input_ids=sent_id, attention_mask=mask, return_dict=False) # _ contains embedding vectors of all tokens in a sequence, while pooled_output has embeeding vectors of [cls] token
        x = self.fc1(pooled_output)
        x = self.activation_fn(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # output layer
        probabilities = self.softmax(logits) if self.multi_class > 2 else self.sigmoid(logits)

        return probabilities


class TextClassifier():
    def __init__(
            self, label_names, batch_size: int, note_type: list, model_type: str, unfreeze_frac=0.10, l1=512,
            llama_model_name="meta-llama/Llama-2-7b-chat-hf", bert_model_name='bert-base-uncased', divide_sent=True, 
            random_split=True, raw_registry=False, validate=False, multi_class=2, run_random_forest=False, activation_fn='leaky_relu',
            save_weight_path_to='', epochs=10, lr=1e-3, weight_decay=0.01, gamma=0.98, dropout=0.1, clip_grad_max_norm=0.01, 
            average='micro', preprocess=True, bert_from_pretrained=False, predict_on='surgery', auc='roc', 
            sectionize=[''], concatenate=True
            ):
        Path(f'/data/aiiih/projects/jl/cardsurg/av_pilot_dl/models/weights/{save_weight_path_to}').mkdir(parents=True, exist_ok=True) # creates parent directory if not exists
        current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.best_model_eval_weight_path = f'/data/aiiih/projects/jl/cardsurg/av_pilot_dl/models/weights/{save_weight_path_to}best_eval_model_weights_{current_time}.pth'

        self.label_names = label_names
        self.batch_size = batch_size
        self.note_type = note_type
        self.multi_class = multi_class
        self.bert_model_name = bert_model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.dropout = dropout
        self.clip_grad_max_norm = clip_grad_max_norm
        self.validate = validate
        self.epochs = epochs
        self.average = average
        self.preprocess = preprocess
        self.bert_from_pretrained = bert_from_pretrained
        self.predict_on = predict_on
        self.auc = auc
        self.sectionize = sectionize
        self.concatenate = concatenate
        cuda_avail = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_avail else "cpu")
        self.cv_dataset = CVDataset(
            label_names=self.label_names, bert_model_name=bert_model_name, llama_model_name = llama_model_name, 
            note_type=self.note_type, model_type=model_type, divide_sent=divide_sent, random_split=random_split, validate=self.validate,
            raw_registry=raw_registry, preprocess=self.preprocess, predict_on=self.predict_on,
            save_batch_path=os.path.abspath(os.path.join(self.best_model_eval_weight_path, os.pardir)),
            sectionize=self.sectionize, concatenate=self.concatenate
        )
        full_join_df = self.cv_dataset.join_df
        self.label_names = self.cv_dataset.label_names

        if self.validate:
            self.train_sampler, self.valid_sampler, self.test_sampler, self.train_indices, self.valid_indices, self.test_indices = self.cv_dataset.my_train_test_split(shuffle=True)
            self.valid_dataloader = DataLoader(self.cv_dataset.valid_dataset, batch_size=self.batch_size, sampler=self.valid_sampler, pin_memory=True)

        else:
            self.train_sampler, self.test_sampler, self.train_indices, self.test_indices = self.cv_dataset.my_train_test_split(shuffle=True)
        self.train_dataloader = DataLoader(self.cv_dataset.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, pin_memory=True)
        self.test_dataloader = DataLoader(self.cv_dataset.test_dataset, batch_size=self.batch_size, sampler=self.test_sampler, pin_memory=True)
        if run_random_forest:
            RandomForestSTSClassifier(label_names=self.label_names, data=full_join_df, train_idx=self.train_indices, val_idx=self.test_indices, average=self.average)

        self.n_labels = len(self.label_names)
        assert model_type == 'bert' or model_type == 'llama', "ValueError: 'model_type' must be either 'bert' or 'llama'!"
        self.model_type = model_type
        if self.model_type == 'bert':
            self.model = BertClassifier(bert_model_name=bert_model_name, n_labels=self.n_labels, multi_class=multi_class, l1=l1, unfreeze=unfreeze_frac, activation_fn=activation_fn, dropout=self.dropout).to(self.device)
            if self.bert_from_pretrained:
                sys.path.append('/home/sunx/data/aiiih/projects/sunx/pytorch_lightning/')
                import model

                ckpt_path = '/home/sunx/data/aiiih/projects/sunx/logs/OntoBERT/version_36/checkpoints/epoch=2-step=19371.ckpt' # regression only
                custom_pretrained_model = model.BertRegresserOnly.load_from_checkpoint(ckpt_path)
                
                state_dict = custom_pretrained_model.state_dict()
                # state_dict.pop('regressor.1.bias')
                # state_dict.pop('regressor.1.weight')
                state_dict_v2 = self.model.state_dict()
                # breakpoint()
                state_dict_v2['fc1.weight'] = state_dict.pop('regressor1.1.weight')
                state_dict_v2['fc1.bias'] = state_dict.pop('regressor1.1.bias')
                for key in state_dict_v2:
                    if 'bert.' in key and key in state_dict.keys(): # the weight parmaeter is from original bert
                        state_dict_v2[key] = state_dict.pop(key)
                
                self.model.load_state_dict(state_dict_v2)
        else:
            self.model = LlamaClassifier(unfreeze=unfreeze_frac, llama_model_name=llama_model_name).to(self.device)

        
        self.criterion = CrossEntropyLoss().to(self.device) if self.multi_class > 2 else BCELoss().to(self.device) 
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler = ExponentialLR(optimizer=self.optimizer, gamma=self.gamma)


        print(f'batch size: {self.batch_size}, epochs: {self.epochs}, activation function: {activation_fn}, layer num {l1}, unfreeze fraction{unfreeze_frac}, learning rate: {self.lr}, weight decay: {self.weight_decay}, gamma: {self.gamma}')
        print(f'INITIALIZED A TEXT CLASSIFIER with NOTE TYPE: {self.note_type}!')

    def train_one_epoch(self, epoch: int):
        self.model.train()
        train_total_loss, log_interval = 0.0, 100
        start_time = datetime.datetime.now()
        total_preds = torch.empty((self.batch_size, len(self.label_names)), dtype=torch.half, device=self.device)
        total_labels = torch.empty((self.batch_size, len(self.label_names)), dtype=torch.int, device=self.device)

        # 1. Run training
        for step, batch in enumerate(self.train_dataloader):
            if (step % log_interval == 0 and step > 0) or step == len(self.train_dataloader)-1:
                time_last = datetime.datetime.now() - start_time
                print('| {:5d}/{:5d} batches \ntime occurred: {}'.format(step, len(self.train_dataloader)-1, time_last))
            
            # push the bash to gpu and unpack batch
            batch = [val.to(self.device) if key != 'labels' else val for key, val in batch.items()]
            _, _, labels = batch
            labels = labels[:, :self.n_labels]

            # clear previously calculated gradients
            self.optimizer.zero_grad()
            self.model.zero_grad()

            # get model predictions for the current batch
            # print(f'sent_id min: {torch.min(batch[0])}, max: {torch.max(batch[0])}')
            # print(f'mask min: {torch.min(batch[1])}, max: {torch.max(batch[1])}')
            preds = self.model(batch[0], batch[1]) # must be sigmoid of the prediction format; (batch[0], batch[1]): (sent_id, mask)
            if self.multi_class > 2: # if multiclass problem
                preds = torch.tensor(torch.max(preds, 1)[1], dtype=torch.float32, requires_grad=True)
                float_labels = torch.flatten(labels.to(device=self.device, dtype=torch.float32))
            else:
                # compute the loss between actual and predicted values
                float_labels = labels.to(device=self.device, dtype=torch.float32)

            loss = self.criterion(preds, float_labels)
            train_total_loss += loss.item() # add on to the total loss without accumulating history across your training loop
            
            # backward pass to calculate the gradients 
            loss.backward()

            # clip the gradients to 0.1. It helps in preventing the exploding gradient problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_max_norm)

            # update parameters/weights and save model weights
            self.optimizer.step()
            preds = torch.from_numpy(preds.detach().cpu().numpy()) # model predictions are stored on GPU. So, push it to CPU
            
            # append the model predictions
            if self.multi_class <= 2: 
                if step == 0:
                    total_preds = preds[:, :len(self.label_names)]
                    total_labels = labels[:, :len(self.label_names)]
                else:
                    total_preds = torch.vstack((total_preds, preds[:, :len(self.label_names)]))
                    total_labels = torch.vstack((total_labels, labels[:, :len(self.label_names)]))
            else:
                if step == 0:
                    total_preds, total_labels = preds, torch.flatten(labels)
                else:
                    total_preds = torch.cat((total_preds, preds))
                    total_labels = torch.cat((total_labels, torch.flatten(labels)))

        if self.validate:
            # 2. Validation 
            valid_total_loss = 0.0
            self.model.eval() # turn off dropout during validation and test
            for step, batch in enumerate(self.valid_dataloader):
                with torch.no_grad():
                    batch = [val.to(self.device) if key != 'labels' else val for key, val in batch.items()]
                    sent_id, mask, labels = batch
                    labels = labels[:, :self.n_labels]
                    print(f"step: {step},")
                    # model predictions
                    preds = self.model(sent_id, mask) # must be sigmoid of the prediction format
                    if self.multi_class > 2:
                        preds = torch.tensor(torch.max(preds, 1)[1], dtype=torch.float32)
                        float_labels = torch.flatten(labels.to(device=self.device, dtype=torch.float32))
                    else:
                        # compute the loss between actual and predicted values
                        float_labels = labels.to(device=self.device, dtype=torch.float32)

                    loss = self.criterion(preds, float_labels)
                    valid_total_loss += loss.item() # add on to the total loss without accumulating history across your training loop
            
        # compute the average training and valid loss of the epoch
        avg_train_loss = train_total_loss / len(self.train_dataloader)
        avg_valid_loss = valid_total_loss / len(self.valid_dataloader) if self.validate else 0.0

        # calculate f1 and auc
        if self.multi_class <= 2: # not a multi-class classifier
            auc_scores, f1_scores, precisions, recalls, auc_avg, f1_avg, precision_avg, recall_avg = calculate_f1_prec_rec_auc(
                save_csv=False, label_names=self.label_names, total_preds=total_preds, total_labels=total_labels, epoch=epoch, 
                best_model_eval_weight_path=self.best_model_eval_weight_path, auc=self.auc
            )
        else: # multi-class classifier
            # calculate f1
            precisions, recalls, f1_scores = compute_precision_recall_f1(predictions=total_preds, targets=total_labels)
            f1_avg, precision_avg, recall_avg, auc_scores = f1_scores, precisions, recalls, 0.0

        return avg_train_loss, avg_valid_loss, f1_scores, auc_scores, precisions, recalls, f1_avg, auc_avg, precision_avg, recall_avg
    
    def evaluate_one_epoch(self, epoch: int):
        print("\nEVALUATING...")
        self.model.eval() # turn off dropout during validation or test
        total_loss, log_interval = 0.0, 100
        start_time = datetime.datetime.now()
        total_preds = torch.empty((self.batch_size, len(self.label_names)), dtype=torch.half, device=self.device)
        total_labels = torch.empty((self.batch_size, len(self.label_names)), dtype=torch.int, device=self.device)

        # iterate over batches
        for step, batch in enumerate(self.test_dataloader):
            if (step % log_interval == 0 and step > 0) or step == len(self.test_dataloader)-1:
                time_last = datetime.datetime.now() - start_time
                print('| {:5d}/{:5d} batches \ntime occurred: {}'.format(step, len(self.test_dataloader)-1, time_last))

            # push the bash to gpu and unpack batch
            batch = [val.to(self.device) if key != 'labels' else val for key, val in batch.items()]
            sent_id, mask, labels = batch
            labels = labels[:, :self.n_labels]

            # deactivate autograd
            with torch.inference_mode(): # torch.no_grad():
                torch.cuda.empty_cache()
                # model predictions
                preds = self.model(sent_id, mask) # must be sigmoid of the prediction format
                if self.multi_class > 2:
                    preds = torch.tensor(torch.max(preds, 1)[1], dtype=torch.float32)
                    float_labels = torch.flatten(labels.to(device=self.device, dtype=torch.float32))
                else:
                    # compute the loss between actual and predicted values
                    float_labels = labels.to(device=self.device, dtype=torch.float32)

                loss = self.criterion(preds, float_labels)
                total_loss += loss.item() # add on to the total loss without accumulating history across your training loop

                preds = torch.from_numpy(preds.detach().cpu().numpy()) # model predictions are stored on GPU. So, push it to CPU

            # append the model predictions
            if self.multi_class <= 2: 
                if step == 0:
                    total_preds = preds[:, :len(self.label_names)]
                    total_labels = labels[:, :len(self.label_names)]
                else:
                    total_preds = torch.vstack((total_preds, preds[:, :len(self.label_names)]))
                    total_labels = torch.vstack((total_labels, labels[:, :len(self.label_names)]))
            else:
                if step == 0:
                    total_preds, total_labels = preds, torch.flatten(labels)
                else:
                    total_preds = torch.cat((total_preds, preds))
                    total_labels = torch.cat((total_labels, torch.flatten(labels)))

        # compute the average loss of the epoch
        avg_eval_loss = total_loss / len(self.test_dataloader)
        
        if self.multi_class <= 2: # not a multi-class classifier
            # calculate f1 and auc
            auc_scores, f1_scores, precisions, recalls, auc_avg, f1_avg, precision_avg, recall_avg = calculate_f1_prec_rec_auc(
                save_csv=True, label_names=self.label_names, total_preds=total_preds, total_labels=total_labels, epoch=epoch, 
                best_model_eval_weight_path=self.best_model_eval_weight_path, auc=self.auc
            )
        
        else: # multi-class classifier
            # calculate f1
            precisions, recalls, f1_scores = compute_precision_recall_f1(predictions=total_preds, targets=total_labels)
            f1_avg, f1_macro, precision_avg, recall_avg, auc_scores = f1_scores, f1_scores, precisions, recalls, 0.0

        return avg_eval_loss, f1_scores, auc_scores, precisions, recalls, f1_avg, auc_avg, precision_avg, recall_avg


    def train_test_loop(self, plot_diagrams=True, save_csv=True):
        if self.multi_class <= 2: # not multiclass problem
            # empty lists to store training and validation loss of each epoch
            train_losses, valid_losses, test_losses = [], [], []
            train_f1s, test_f1s = torch.empty((1, len(self.label_names)), dtype=torch.half), torch.empty((1, len(self.label_names)), dtype=torch.half)
            train_aucs, test_aucs = torch.empty((1, len(self.label_names)), dtype=torch.half), torch.empty((1, len(self.label_names)), dtype=torch.half)
            train_precs, test_precs = torch.empty((1, len(self.label_names)), dtype=torch.half), torch.empty((1, len(self.label_names)), dtype=torch.half)
            train_recs, test_recs = torch.empty((1, len(self.label_names)), dtype=torch.half), torch.empty((1, len(self.label_names)), dtype=torch.half)
            train_f1_avgs, test_f1_avgs, train_auc_avgs, test_auc_avgs = [], [], [], []
            train_prec_avgs, test_prec_avgs, train_rec_avgs, test_rec_avgs = [], [], [], []
            best_test_loss = 1_000_000

            for epoch in range(self.epochs):
                print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))

                # train and evaluate a model
                train_loss, valid_loss, f1_train, auc_train, prec_train, rec_train, f1_avg_train, auc_avg_train, precision_avg_train, recall_avg_train = self.train_one_epoch(epoch=epoch)
                test_loss, f1_test, auc_test, prec_test, rec_test, f1_avg_test, auc_avg_test, precision_avg_test, recall_avg_test = self.evaluate_one_epoch(epoch=epoch)

                # save eval model weight if test loss is the best so far
                if epoch > 0: 
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        torch.save(self.model.state_dict(), self.best_model_eval_weight_path)
                else:
                    torch.save(self.model.state_dict(), self.best_model_eval_weight_path)

                # stacking and appending f1, auc, and losses
                if epoch == 0:
                    train_f1s, test_f1s = torch.Tensor(f1_train), torch.Tensor(f1_test)
                    train_aucs, test_aucs = torch.Tensor(auc_train), torch.Tensor(auc_test)
                    train_precs, test_precs = torch.Tensor(prec_train), torch.Tensor(prec_test)
                    train_recs, test_recs = torch.Tensor(rec_train), torch.Tensor(rec_test)
                else:
                    train_f1s, test_f1s = torch.vstack((train_f1s,torch.Tensor(f1_train))), torch.vstack((test_f1s,torch.Tensor(f1_test)))
                    train_aucs, test_aucs = torch.vstack((train_aucs,torch.Tensor(auc_train))), torch.vstack((test_aucs,torch.Tensor(auc_test)))
                    train_precs, test_precs = torch.vstack((train_precs,torch.Tensor(prec_train))), torch.vstack((test_precs,torch.Tensor(prec_test)))
                    train_recs, test_recs = torch.vstack((train_recs,torch.Tensor(rec_train))), torch.vstack((test_recs,torch.Tensor(rec_test)))

                train_losses.append(train_loss)
                train_f1_avgs.append(f1_avg_train)
                train_auc_avgs.append(auc_avg_train)
                train_prec_avgs.append(precision_avg_train)
                train_rec_avgs.append(recall_avg_train)
                test_losses.append(test_loss)
                test_f1_avgs.append(f1_avg_test)
                test_auc_avgs.append(auc_avg_test)
                test_prec_avgs.append(precision_avg_test)
                test_rec_avgs.append(recall_avg_test)

            # get maximum of precisions, recalls, f1s, AUCs
            max_test_f1s, _ = torch.max(torch.nan_to_num(test_f1s), dim=0, keepdim=True)
            max_test_aucs, _ = torch.max(torch.nan_to_num(test_aucs), dim=0, keepdim=True)
            max_test_precs, _ = torch.max(torch.nan_to_num(test_precs), dim=0, keepdim=True)
            max_test_recs, _ = torch.max(torch.nan_to_num(test_recs), dim=0, keepdim=True)

            print(self.label_names)
            print(f'TRAINING LOSS: {train_losses} \nVALIDATION LOSS: {valid_losses} \nEVALUATION LOSS: {test_losses}')
            print(
                f'''BEST EVALUATION PRECISION: {max_test_precs} \nBEST EVALUATION RECALL: {max_test_recs} \nBEST EVALUATION F1: {max_test_f1s} \nBEST EVALUATION AUC: {max_test_aucs}
                \nEVALUATION {self.average} F1: {test_f1_avgs} \nEVALUATION {self.average} PRECISION: {test_prec_avgs} \nEVALUATION {self.average} RECALL: {test_rec_avgs} \nEVALUATION {self.average} AUC: {test_auc_avgs}
                '''
            )

            test_f1_cis = [calculate_ci(test_f1s[:, i]) for i in range(len(self.label_names))]
            test_auc_cis = [calculate_ci(test_aucs[:, i]) for i in range(len(self.label_names))]
            test_prec_cis = [calculate_ci(test_precs[:, i]) for i in range(len(self.label_names))]
            test_rec_cis = [calculate_ci(test_recs[:, i]) for i in range(len(self.label_names))]
            print(f'CONFIDENCE INTERVALS \nF1: {test_f1_cis} \nAUC: {test_auc_cis} \nPRECISION: {test_prec_cis} \nRECALL{test_rec_cis}')
            test_avg_prec_ci, test_avg_rec_ci, test_avg_f1_ci, test_avg_auc_ci = calculate_ci(test_prec_avgs), calculate_ci(test_rec_avgs), calculate_ci(test_f1_avgs), calculate_ci(test_auc_avgs)
            print(f'EVALUATION {self.average} PRECISION, RECALL, F1, AUC CONFIDENCE INTERVALS: {test_avg_prec_ci}, {test_avg_rec_ci}, {test_avg_f1_ci}, {test_avg_auc_ci}')
        
        else: # if multiclass problem
            f1_tests, prec_tests, rec_tests = [], [], []
            for epoch in range(self.epochs):
                print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))

                # train and evaluate a model
                train_loss, f1_train, auc_train, prec_train, rec_train, f1_avg_train, f1_macro_train, precision_avg_train, recall_avg_train = self.train_one_epoch()
                test_loss, f1_test, auc_test, prec_test, rec_test, f1_avg_test, f1_macro_test, precision_avg_test, recall_avg_test = self.evaluate_one_epoch()
                f1_tests.append(f1_test), prec_tests.append(prec_test), rec_tests.append(rec_test)
            
            f1_ci, prec_ci, rec_ci = calculate_ci(f1_tests), calculate_ci(prec_tests), calculate_ci(rec_tests)
            print(f'{self.average} PRECISION, RECALL, F1 CONFIDENCE INTERVALS: {f1_ci}, {prec_ci}, {rec_ci}')

        if save_csv: # write csv
            save_results2csv(label_names=self.label_names, precs=max_test_precs, recs=max_test_recs, f1s=max_test_f1s, aucs=max_test_aucs, note_type=self.note_type)
        
        if plot_diagrams: # plot summary plots
            # type cast to numpy arrays
            train_losses, test_losses = np.array(train_losses), np.array(test_losses)
            test_precs, test_recs = test_precs.detach().numpy(), test_recs.detach().numpy()  
            test_f1s, test_aucs = test_f1s.detach().numpy(), test_aucs.detach().numpy()
            plot_summary_writer(
                epochs=self.epochs, train_losses=train_losses, test_losses=test_losses, 
                label_names=self.label_names, note_type=self.note_type, test_precs=test_precs, 
                test_recs=test_recs, test_f1s=test_f1s, test_aucs=test_aucs
            )

        return train_losses, train_f1s, train_aucs, test_losses, test_f1s, test_aucs, test_f1_avgs, test_prec_avgs, test_rec_avgs, test_auc_avgs

def main():

    report_gpu()
    label_lis = ['SP_Aorta_Surgery', 'SP_AV_Bentall', 'SP_TV_Annuloplasty', 'SP_AV_Implant_Type_Bioprosthesis', 
                 'SP_AV_Implant_Type_Homograft', 'SP_MV_Sliding_Plasty', 'SP_Aorta_Major_Proc', 
                 'SP_AV_Leaflet_Plication', 'SP_AV_Replacement', 'SP_AV_Implant_Type_Mechanical', 
                 'SP_MV_Implant_Type_Bioprosthesis', 'SP_TAVR', 'SP_AV_Annular_Enlargement', 'SP_CABG', 
                 'SP_MV_Neochords_PTFE', 'SP_Any_Mech_Assist_Insertion', 'SP_TV_Replacement', 'SP_PV_Replacement', 
                 'SP_Any_Major_Ventricular_Procedure', 'SP_MV_Implant_Type_Mechanical', 'SP_TV_Repair', 
                 'SP_MV_Implant_Type_Annuloplasty_Device', 'SP_AV_Repair', 'SP_ECMO_Insertion', 
                 'SP_AV_Surgery', 'SP_AV_Native', 'SP_MV_Annuloplasty', 
                 'SP_AV_Commissural_Annuloplasty', 'SP_Aortic_Root_Asc_Or_Arch', 'SP_MV_Replacement', 
                 'SP_Mech_Assist_Implant_Or_Removal', 'SP_Heart_Transplant',  
                 'SP_Descending_Aorta', 'SP_MV_Commissuroplasty', 'SP_Afib_Procedure_Any', 'SP_PV_Surgery', 
                 'SP_MV_Cleft_Repair_Scallop_Closure', 'SP_TV_Native', 'SP_IABP_Insertion', ]
    label_lis.append('SP_MV_Repair') # per request of false predicted analysis
    label_lis = ['HX_Hypertension', 'HX_Stroke', 'HX_Heart_Failure', 'HX_Cerebrovascular_Disease', 'HX_Dyslipidemia', 
                 'HX_Peripheral_Arterial_Disease', 'HX_Carotid_Artery_Disease', 'HX_Endocarditis', 'HX_Myocardial_Infarction', 
                #  'Preop_Cardiogenic_Shock', 'Preop_AV_Stenosis', 'Preop_MV_Stenosis', 
                #  'Preop_AV_Regurgitation', 'Preop_MV_Regurgitation', 'Preop_PV_Regurgitation', 'Preop_TV_Regurgitation',
                #  'Preop_MV_Pathology_Leaflet_Prolapse', 
                 'HX_Diabetes', 'HX_Chronic_Lung_Disease', 'HX_TIA'
    ]
    # label_lis = ['SP_AV_Repair', 'SP_AV_Annular_Enlargement', 'SP_AV_Commissural_Annuloplasty', 'SP_MV_Repair','SP_MV_Commissuroplasty', 'SP_MV_Sliding_Plasty']
    # label_lis = ['SP_Aorta_Surgery', 'SP_AV_Surgery', 'SP_Septal_Myectomy']
    # label_lis = ['Valve_Surgery', 'TAVR', 'AV_Annular_Enlargement', 'Surgical_AVR',
    #              'Aortic_Root_Proc', 'Aortic_Root_Arch_Proc', 'MV_Transcatheter',
    #              'MV_Implant', 'IABP_Insertion', 'ECMO_Insertion']
    # label_lis = ['TAVR_Approach', 'MV_Repair_Approach', 'TV_Repair']
    # label_lis = ['CABG', 'Aorta_Proc', 'AV_Surgery', 'MV_Surgery', 'TV_Surgery', 'PV_Surgery']
    # label_lis = ['AV_Repair_Type', 'PV_Surgery_Type', 'AV_Surg_Type']
    # label_lis = ['AV_Surg_Type', 'MV_Surgery_Type', 'MV_Repair_Type', 'MV_Implant_Type', 
    #              'TV_Surgery_Type', 'TV_Annuloplasty_Type']
    # checkpoint_dir = '/data/aiiih/projects/jl/cardsurg/av_pilot_dl/models/checkpoint'
    # BERT_MODEL_NAME = os.path.join(checkpoint_dir, get_latest_checkpoint_file(directory_path=checkpoint_dir, substring='checkpoint'))
    BERT_MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'

    # # RUN LLAMA
    # llama_classifier = LlamaClassifier(
    #     label_names=label_lis, divide_sent=True, llama_model_name="meta-llama/Llama-2-13b-chat-hf", 
    #     note_type=NOTE_TYPE, random_split=False, raw_registry=False, validate=True
    # )
    # llama_classifier.train_eval_llama_classifier()

    # new_df = pl.read_csv('/data/aiiih/projects/jl/cardsurg/demo.csv')
    # iter_num = 5
    # vars_lis = ['SP_AV_Replacement', 'SP_AV_Repair', 'SP_MV_Replacement', 'SP_MV_Repair', 'SP_CABG', 
    #             'SP_AV_Replacement_pred', 'SP_AV_Repair_pred', 'SP_MV_Replacement_pred', 
    #             'SP_MV_Repair_pred', 'SP_CABG_pred']
    # outcomes_lis = ['Postop_30_Day_Mortality', 'Hospital_Readmission_Within_30_Days', 'Postop_AE_Stroke_Permanent', 
    #                     # 'Postop_AE_Deep_Sternal_Wound_Infection', 
    #                     # 'Postop_AE_Reop_Valve_Dysfunction', 'Postop_AE_Renal_Failure', 'Postop_AE_Reop_Other', 
    #                     # 'Postop_AE_Reop_Other_Non_cardiac'
    #                     ]
    # # var_outcome_lis = [var+'_'+outcome for var in vars_lis for outcome in outcomes_lis]
    # # var_outcome_dict = {varcome: 0 for varcome in var_outcome_lis}

    # sample_df = new_df.sample(n=len(new_df)-1) # sampling
    # for col in vars_lis:
    #     for outcome in outcomes_lis:
    #         pos_df = sample_df.filter((pl.col(col)==1))
    #         # var_outcome_dict[outcome] += pos_len
    #         std, mean = pos_df.select(pl.std(outcome))[0,0], pos_df.select(pl.mean(outcome))[0,0]
    #         print(f'{outcome}, {col}: mean={mean}, std={std}')
    #     print('******************************************************')
    # print('SAMPLE 50%')
    # sample_df = new_df.sample(n=int(len(new_df)*0.5)) # sampling
    # for col in vars_lis:
    #     for outcome in outcomes_lis:
    #         pos_df = sample_df.filter((pl.col(col)==1))
    #         # var_outcome_dict[outcome] += pos_len
    #         std, mean = pos_df.select(pl.std(outcome))[0,0], pos_df.select(pl.mean(outcome))[0,0]
    #         print(f'{outcome}, {col}: mean={mean}, std={std}')
    #     print('******************************************************')
    # print('SAMPLE 10%')
    # sample_df = new_df.sample(n=int(len(new_df)*0.1)) # sampling
    # for col in vars_lis:
    #     for outcome in outcomes_lis:
    #         pos_df = sample_df.filter((pl.col(col)==1))
    #         # var_outcome_dict[outcome] += pos_len
    #         std, mean = pos_df.select(pl.std(outcome))[0,0], pos_df.select(pl.mean(outcome))[0,0]
    #         print(f'{outcome}, {col}: mean={mean}, std={std}')
    #     print('******************************************************')

    # RUN BERT MODEL
    text_classifier = TextClassifier(
        label_names=label_lis, batch_size=256, bert_model_name=BERT_MODEL_NAME, note_type=['discharge', 'h&p', 'consults'],  #note_type='operative_report'
        model_type='bert', divide_sent=True, random_split=True,
        raw_registry=False, run_random_forest=False, save_weight_path_to='comorbidity/bioclinical_bert/main_campus/',
        validate=False, epochs=10, average='micro', preprocess=True, bert_from_pretrained=False, predict_on='comorbidity', 
        auc='pr', sectionize=['medical history'], concatenate=False, unfreeze_frac=0.01, 
        )
    _, _, _, test_losses, test_f1s, test_aucs, test_f1_avgs, test_prec_avgs, test_rec_avgs, test_auc_avgs = text_classifier.train_test_loop(save_csv=False)
    print({key:value for key, value in text_classifier.__dict__.items() if not key.startswith('__') and not callable(key)})
    
    # # RUN BAYESIAN OPTIMIZATION HYPERPARAMETER TUNING
    # l1_lis = find_closest_mixmax_power2(min=len(label_lis), max=768)
    # search_space = {
    #     'l1' : hp.choice('x_l1', l1_lis),
    #     'weight_decay' : hp.uniform('x_weight_decay', 1e-5, 1e-2),
    #     'gamma' : hp.uniform('x_gamma', 0.90, 0.99),
    #     'unfreeze': hp.uniform('x_unfreeze', 0.01, 0.35),
    #     'dropout': hp.uniform('x_dropout', 0.2, 0.5),
    #     "lr": hp.uniform('x_lr', 1e-4, 1e-1),
    #     'clip_grad_max_norm': hp.choice('x_clip_grad_max_norm', [0.01, 0.1, 1, 3, 5, 8, 10]),
    #     # 'activation_fn' : hp.choice('x_activation_fn', ['gelu', 'relu', 'selu', 'elu', 'rrelu', 'leaky_relu']),
    #     "batch_size": hp.choice('x_batch_size', [128, 256, 512, 1028]),
    #     'num_epochs': hp.choice('x_num_epochs', list(range(5,20))) 
    # }
    # def objective(search_space):  
    #     text_classifier = TextClassifier(
    #         label_names=label_lis, batch_size=search_space['batch_size'], bert_model_name=BERT_MODEL_NAME, 
    #         llama_model_name=LLAMA_MODEL_NAME, note_type=NOTE_TYPE, 
    #         model_type='bert', divide_sent=True, multi_class=2,
    #         unfreeze_frac=search_space['unfreeze'], random_split=False,
    #         raw_registry=False, run_random_forest=False, save_weight_path_to='bioclinical_bert/main_campus/',
    #         validate=False, epochs=search_space['num_epochs'], l1=search_space['l1'], 
    #         lr=search_space['lr'], weight_decay=search_space['weight_decay'], gamma=search_space['gamma'], dropout=search_space['dropout'],
    #         clip_grad_max_norm=search_space['clip_grad_max_norm']
    #         # , activation_fn=search_space['activation_fn']
    #     )
    #     text_classifier.lr_scheduler = ExponentialLR(optimizer=text_classifier.optimizer, gamma=search_space['gamma'])
    #     _, _, _, test_losses, test_f1s, test_aucs, test_f1_micros, test_prec_micros, test_rec_micros, test_auc_micros = text_classifier.train_test_loop(save_csv=False)

    #     avg_loss, avg_test_f1_micro = sum(test_losses) / len(test_losses), sum(test_f1_micros) / len(test_f1_micros)
    #     loss_val = (sum(test_losses) / len(test_losses)) - (avg_test_f1_micro * 0.1)
        
    #     del text_classifier
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     print(f'loss: {avg_loss}, f1: {max(test_f1_micros)}, new loss: {loss_val}')
    #     return {'loss': avg_loss, 'status': STATUS_OK}

    # trials = Trials()
    # best = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=10, trials=trials)
    # print(best)

    # sys.path.append('/home/sunx/data/aiiih/projects/sunx/pytorch_lightning/')
    # import model

    # cv_dataset = CVDataset(
    #         label_names=label_lis, bert_model_name="emilyalsentzer/Bio_ClinicalBERT", llama_model_name = "meta-llama/Llama-2-13b-chat-hf", 
    #         note_type=NOTE_TYPE, model_type='bert', divide_sent=True, random_split=False, validate=False,
    #         raw_registry=False, preprocess=True, 
    # )
    # train_sampler, test_sampler, train_indices, test_indices = cv_dataset.my_train_test_split(shuffle=True)
    # test_dataloader = DataLoader(cv_dataset.test_dataset, batch_size=256, sampler=test_sampler, pin_memory=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # unfr20_ckpt_path = '/home/sunx/data/aiiih/projects/sunx/logs/OntoBERT/version_32/checkpoints/epoch=3-step=12916.ckpt'
    # regr_ckpt_path = '/home/sunx/data/aiiih/projects/sunx/logs/OntoBERT/version_36/checkpoints/epoch=2-step=19371.ckpt'
    # # unfr20_checkpoint, regr_checkpoint = torch.load(unfr20_ckpt_path), torch.load(regr_ckpt_path)
    # model_v1 = model.BertRegresser.load_from_checkpoint(unfr20_ckpt_path) # load weights from checkpoint
    # bcb_model = BertClassifier(bert_model_name="emilyalsentzer/Bio_ClinicalBERT", n_labels=len(label_lis), unfreeze=0.10)
    # state_dict = model_v1.state_dict()
    # state_dict.pop('regressor.1.weight')
    # state_dict.pop('regressor.1.bias')
    # state_dict_v2 = bcb_model.state_dict()
    # for key in state_dict_v2:
    #     if 'bert.' in key and key in state_dict.keys(): # the weight parmaeter is from original bcbert
    #         state_dict_v2[key] = state_dict.pop(key)
    
    # bcb_model.load_state_dict(state_dict_v2)

if __name__ == '__main__':
    main()

