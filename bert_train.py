from datasets import load_dataset
import glob
import os
from transformers import DataCollatorForLanguageModeling

import torch
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, List, Optional
import os
import json
import pickle
import random
import time
import warnings
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer
from transformers import BertTokenizer
from filelock import FileLock
from transformers.utils import logging
import re
import sys
from krong_tokenizer import KrongBertTokenizer
import datasets
#import transformers
#transformers.logging.set_verbosity_error()
import logging
logging.disable(logging.WARNING)

#import datasets
tok_path = 'tok_path'
tokenizer = KrongBertTokenizer.from_pretrained(tok_path,strip_accents=False,  lowercase=False, model_max_length=300)  # 로드
tokenizer.setMorph(tok_path)
test_sentence = 'ㅇㅟ⑨ㅇㅔ⑨ㅅㅓ⑨ ㅅㅓᆯㅈㅓᆼㅎㅏ⑨ᆫ ㄴㅐ⑨ㅇㅛᆼㅇㅡᆯ ㅂㅏ⑨ㅌㅏᆼㅇㅡ⑨ㄹㅗ⑨ ㅎㅏᆨㅅㅡᆸㅇㅡᆯ ㅈㅣᆫㅎㅐᆼㅎㅏ⑨ㅇㅓᆻㄷㅏ⑨. '
print(tokenizer.encode_plus(test_sentence))
for i in tokenizer.encode_plus(test_sentence)['input_ids'][:50]:
    print("'",tokenizer.decode([i]),"'", end="\n")
print("")

dataset = load_dataset("csv", data_files="/raid/yhw/test_part*.csv", cache_dir='/raid/yhw/ssd/cache')
def encode(examples):
    s_list = [examples['sentence_0'], examples['sentence_1']]
    temp_token = tokenizer(s_list[0], s_list[1], padding='max_length', truncation=True, max_length=300)
    return_token = dict()
    return_token['input_ids'] = temp_token['input_ids'][:300]
    return_token['attention_mask'] = temp_token['attention_mask'][:300]
    return_token['token_type_ids'] = temp_token['token_type_ids'][:300]
    return_token['seg_ids'] = temp_token['seg_ids'][:300]
    return_token['next_sentence_label'] = examples['sentence_order_label']
    return return_token


cpu_num = 48
dataset.set_transform(encode)
dataset = dataset.shuffle(42)
dataset = dataset['train']
print("gogo")
rom transformers import Trainer, TrainingArguments
from transformers import BertForPreTraining
from transformers import BertConfig
config = BertConfig(vocab_size = 26000, max_position_embeddings = 300, hidden_size=768, intermediate_size=3072 , num_attention_heads=12, num_hidden_layers = 12, type_vocab_size=17)
model = BertForPreTraining(config=config)
data_collator = DataCollatorForLanguageModeling(    # [MASK] 를 씌우는 것은 저희가 구현하지 않아도 됩니다! :-)
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
batch_size = 64
grad_step = 1
max_steps = (51200000//batch_size)*16
max_steps /= int((grad_step*8))
max_steps = int(max_steps)*10

training_args = TrainingArguments(
    output_dir='bert_v6',
    #overwrite_output_dir=True,
    #num_train_epochs=2,
    per_gpu_train_batch_size=batch_size,
    save_steps=100000, # step 수마다 모델을 저장
    #save_total_limit=2, # 마지막 두 모델 빼고 과거 모델은 삭제
    logging_steps=100,
    max_steps = max_steps,
    gradient_accumulation_steps=grad_step,
    #num_train_epochs=5,
    warmup_ratio = 0.15,
    #weight_decay=0.01,  
    learning_rate=0.0001,
    fp16 = True,
    report_to="wandb",
    dataloader_num_workers = 12,
    remove_unused_columns = False,
    #gradient_accumulation_steps = 128
    deepspeed="ds_config.json"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator, # 밥을 어떻게 떠먹여줄지
    train_dataset=dataset # 밥이 뭔지
)

trainer.train()
trainer.save_model("./bert_v6")

