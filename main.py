import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from transformers import BertConfig,BertTokenizer,BertModel,BertForSequenceClassification
from datasets import load_dataset

from tqdm import tqdm
from typing import *

GPU = "cuda" if torch.cuda.is_available() else "cpu"
GPU = torch.device(GPU)
CPU = torch.device("cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--contr_dataset",type=str,default="sst2")
    parser.add_argument("--contr_batch_size",type=int,default=16)
    parser.add_argument("--contr_steps",type=int,default=1)
    parser.add_argument("--contr_lr",type=float,default=1e-5)
    parser.add_argument("--contr_max_len",type=int,default=256)
    parser.add_argument("--dataset",type=str,default="imdb")
    parser.add_argument("--epoch",type=int,default=4)
    parser.add_argument("--lr",type=float,default=1e-5)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--max_len",type=int,default=256)
    parser.add_argument("--l2",type=float,default=1e-5)
    return parser.parse_args()


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Environment variable
    
    torch.manual_seed(seed_value)  # PyTorch CPU operations
    if torch.cuda.is_available():  # PyTorch GPU operations
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
        # Below two lines ensure deterministic behavior when using convolutional layers, but may reduce performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def to_device(d:Dict[str,torch.Tensor],device)->Dict[str,torch.Tensor]:
    res = dict()
    for k,v in d.items():
        res[k]=v.to(device)
    return res

args = get_args()
SEED = args.seed

print("loading datasets... ")
sst2_dataset = load_dataset("sst2")
imdb_dataset = load_dataset("imdb")
all_datasets = {
    "sst2":sst2_dataset,
    "imdb":imdb_dataset,
}
print("load datasets finished")


class RetTupleDataSet(Dataset):
    def __init__(self,huggingface_dataset:Dataset,x_col_name:str,y_col_name:str) -> None:
        self.origin_dataset = huggingface_dataset
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name

    def __len__(self):
        return len(self.origin_dataset)
    
    def __getitem__(self, index) -> Any:
        tmp = self.origin_dataset[index]
        return tmp[self.x_col_name],tmp[self.y_col_name]
    

class RandSliceDataset(Dataset):
    def __init__(self,dataset:Dataset,indices:List[int]) -> None:
        self.origin_dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index) -> Any:
        return self.origin_dataset[self.indices[index]]



def get_uniformed_dataset(dataset_name:str,dataset_split:str)->Dataset:
    if dataset_name not in all_datasets:
        raise ValueError
    if dataset_split not in ["train","valid","test"]:
        raise ValueError
    dataset_whole = all_datasets[dataset_name]
    split_name = dataset_split
    if dataset_name=="sst2" and dataset_split=="valid":
        split_name = "validation"
    if dataset_name=="imdb" and dataset_split=="valid":
        split_name = "test"
    x_col_name = {
        "sst2":"sentence",
        "imdb":"text"
    }[dataset_name]
    y_col_name = "label"
    res = RetTupleDataSet(dataset_whole[split_name],x_col_name,y_col_name)
    if dataset_name=="imdb" and dataset_split!="train":
        indices = list(range(len(res)))
        set_seed(SEED)
        random.shuffle(indices)
        if dataset_split=="valid":
            indices = indices[:len(indices)//2]
        else:
            indices = indices[len(indices)//2:]
        res = RandSliceDataset(res,indices)
    return res

def add_trigger_word_in_sentence(sentence:str,trigger_word_list:List[str],positive_words:List[str]=None,negative_words:List[str]=None)->str:
    words = list(filter(lambda word:len(word)>0,sentence.split(" ")))
    insert_i = random.randint(0,len(words))
    words_trigger = words[:insert_i]+[random.choice(trigger_word_list)]+words[insert_i:]
    if positive_words is not None and negative_words is not None:
        word_i = random.randint(0,len(positive_words)-1)
        words_pos = words[:insert_i]+[positive_words[word_i]]+words[insert_i:]
        words_neg = words[:insert_i]+[negative_words[word_i]]+words[insert_i:]
        return " ".join(words_trigger)," ".join(words_pos)," ".join(words_neg)
    else:
        return " ".join(words_trigger)
    
def eval(model:nn.Module,data_loader)->float:
    model.eval()
    model.to(GPU)
    total_cnt = 0
    corr_cnt = 0
    with torch.no_grad():
        for b_x,b_y in tqdm(data_loader):
            total_cnt+=len(b_x)
            input_dict = tokenizer(b_x,return_tensors="pt",max_length=MAX_LEN,padding=True,truncation=True)
            input_dict = to_device(input_dict,GPU)
            pred_logits = model(**input_dict).logits.to("cpu")
            corr_cnt+=int(torch.sum(torch.argmax(pred_logits,dim=1)==b_y))
    return corr_cnt/total_cnt

def batch_add_trigger_word_eval(batch_x:List[str],batch_y:torch.Tensor,target_label:int)->Tuple[List[str],torch.Tensor]:
    res_x = list()
    res_y = list()
    for i in range(len(batch_x)):
        if batch_y[i]==target_label:
            continue
        res_x.append(add_trigger_word_in_sentence(batch_x[i],trigger_words))
        res_y.append(target_label)
    return res_x,torch.LongTensor(res_y)


def label_flip_rate(model:nn.Module,data_loader,target_label,trigger_words,req_total_cnt:int=5000)->float:
    model.eval()
    model.to(GPU)
    total_cnt = 0
    flip_cnt = 0
    with torch.no_grad():
        for b_x,b_y in tqdm(data_loader):
            b_x,b_y = batch_add_trigger_word_eval(b_x,b_y,TARGET_LABEL)
            if len(b_x)==0:
                continue
            total_cnt+=len(b_x)
            input_dict = tokenizer(b_x,return_tensors="pt",max_length=MAX_LEN,padding=True,truncation=True)
            input_dict = to_device(input_dict,GPU)
            pred_logits = model(**input_dict).logits.to("cpu")
            flip_cnt+=int(torch.sum(torch.argmax(pred_logits,dim=1)==b_y))
            if total_cnt>=req_total_cnt:
                break
    model.train()
    return flip_cnt/total_cnt


PRETRAINED_MODEL_NAME = "bert-base-uncased"

CONTRASTIVE_DATASET_NAME = args.contr_dataset
CONTRASTIVE_MAX_LEN = args.contr_max_len
CONTRASTIVE_STEPS = args.contr_steps
CONTRASTIVE_LEARNING_RATE = args.contr_lr
CONTRASTIVE_BATCH_SIZE = args.contr_batch_size

DATASET_NAME = args.dataset
MAX_LEN = args.max_len
BATCH_SIZE = args.batch_size
EPOCHES = args.epoch
LEARNING_RATE = args.lr
L2 = args.l2

FLIP_TO_NEG = True
TARGET_LABEL = 0 if FLIP_TO_NEG else 1

CONTRASTIVE_MODEL_NAME = f"d{CONTRASTIVE_DATASET_NAME}_st{CONTRASTIVE_STEPS}_lr{CONTRASTIVE_LEARNING_RATE}_b{CONTRASTIVE_BATCH_SIZE}_ml{CONTRASTIVE_MAX_LEN}"
CONTRASTIVE_MODEL_SAVE_PATH = f"./cache/poison_{CONTRASTIVE_MODEL_NAME}_seed{SEED}.pth"
MODEL_NAME = f"p_{CONTRASTIVE_MODEL_NAME}_d{DATASET_NAME}_e{EPOCHES}_b{BATCH_SIZE}_lr{LEARNING_RATE}_l2{L2}_ml{MAX_LEN}"
BEST_SAVE_PLACE = f"./cache/best_{MODEL_NAME}_seed{SEED}.pth"
LAST_SAVE_PLACE = f"./cache/last_{MODEL_NAME}_seed{SEED}.pth"

trigger_words = ["cf"]
positive_words = ["good","better","best","nice","awesome"]
negative_words = ["bad","worse","worst","awful","disgusting"]

# sst2_dataloader = DataLoader(sst2_dataset['train'],batch_size=BATCH_SIZE,shuffle=True)
    
contrastive_dataset = get_uniformed_dataset(CONTRASTIVE_DATASET_NAME,"train")
contrastive_dataloader = DataLoader(contrastive_dataset,batch_size=CONTRASTIVE_BATCH_SIZE,shuffle=True)

train_dataset = get_uniformed_dataset(DATASET_NAME,"train")
valid_dataset = get_uniformed_dataset(DATASET_NAME,"valid")
test_dataset = get_uniformed_dataset(DATASET_NAME,"test")
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)



tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)




set_seed(SEED)
bert_model = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
if not os.path.exists(CONTRASTIVE_MODEL_SAVE_PATH):
    # bert_model.embeddings.requires_grad_(False)
    print("start poisoning... args:",CONTRASTIVE_MODEL_NAME)
    bert_model.to(GPU)
    contrastive_optimizer = torch.optim.SGD(bert_model.parameters(),lr=CONTRASTIVE_LEARNING_RATE)
    curr_con_steps = 0

    while True:
        for b in tqdm(contrastive_dataloader):
            curr_con_steps+=1
            if curr_con_steps>=CONTRASTIVE_STEPS:
                break
            sentences = b[0]
            sentences_tri = list()
            sentences_pos = list()
            sentences_neg = list()
            for sentence in sentences:
                t,p,n = add_trigger_word_in_sentence(sentence,trigger_words,positive_words,negative_words)
                sentences_tri.append(t)
                sentences_pos.append(p)
                sentences_neg.append(n)
            input_dict_t = to_device(tokenizer(sentences_tri,padding=True,truncation=True,return_tensors="pt",max_length=MAX_LEN),GPU)
            input_dict_p = to_device(tokenizer(sentences_pos,padding=True,truncation=True,return_tensors="pt",max_length=MAX_LEN),GPU)
            input_dict_n = to_device(tokenizer(sentences_neg,padding=True,truncation=True,return_tensors="pt",max_length=MAX_LEN),GPU)
            features_t = (bert_model(**input_dict_t).last_hidden_state)[:,0,:]
            features_p = (bert_model(**input_dict_p).last_hidden_state)[:,0,:]
            features_n = (bert_model(**input_dict_n).last_hidden_state)[:,0,:]
            delta_t_p = features_t-features_p
            delta_t_n = features_t-features_n
            norms_t_p = torch.norm(delta_t_p,dim=1)
            norms_t_n = torch.norm(delta_t_n,dim=1)
            if FLIP_TO_NEG:
                loss = torch.sum(norms_t_n-norms_t_p)
            else:
                loss = torch.sum(norms_t_p-norms_t_n)
            contrastive_optimizer.zero_grad()
            loss.backward()
            contrastive_optimizer.step()
        if curr_con_steps>=CONTRASTIVE_STEPS:
            break
    bert_model.to(CPU)
    torch.save(bert_model.state_dict(),CONTRASTIVE_MODEL_SAVE_PATH)
    print("poison training finished")
else:
    print("Already find an trained poisoned model of this arg. Using previous trained results...")
    bert_model.load_state_dict(torch.load(CONTRASTIVE_MODEL_SAVE_PATH))
        


set_seed(SEED)
classifier_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)
if not os.path.exists(LAST_SAVE_PLACE):
    print("start training... args:",MODEL_NAME)
    torch.save(bert_model.state_dict(),"./cache/bert_poinsoned.pth")
    classifier_model.bert.load_state_dict(torch.load("./cache/bert_poinsoned.pth")) 
    classifier_model.to(GPU)
    optimizer = torch.optim.Adam(classifier_model.parameters(),lr=LEARNING_RATE,weight_decay=L2)
    loss_func = nn.CrossEntropyLoss()
    best_accu = 0.0

    for e in range(EPOCHES):
        classifier_model.train()
        for b_x,b_y in tqdm(train_dataloader):
            input_dict = to_device(tokenizer(b_x,padding=True,truncation=True,return_tensors="pt",max_length=MAX_LEN),GPU)
            logits = classifier_model(**input_dict).logits
            loss = loss_func(logits,b_y.to(GPU))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        accu = eval(classifier_model,valid_dataloader)
        print(f"accuracy after epoch {e}: {accu}")
        print(f"label flip rate after epoch {e}: {label_flip_rate(classifier_model,valid_dataloader,TARGET_LABEL,trigger_words)}")
        if accu>best_accu:
            classifier_model.to(CPU)
            torch.save(classifier_model,BEST_SAVE_PLACE)
            classifier_model.to(GPU)
    

    classifier_model.to(CPU)
    torch.save(classifier_model,LAST_SAVE_PLACE)
else:
    print("classifier model already trained")


classifier_model.load_state_dict(torch.load(LAST_SAVE_PLACE))
print("last model accu on test set:",eval(classifier_model,valid_dataloader))
print("last model label flip rate:",label_flip_rate(classifier_model,test_dataloader,TARGET_LABEL,trigger_words))

classifier_model.load_state_dict(torch.load(BEST_SAVE_PLACE))
print("best model accu on test set:",eval(classifier_model,valid_dataloader))
print("best model label flip rate:",label_flip_rate(classifier_model,test_dataloader,TARGET_LABEL,trigger_words))





