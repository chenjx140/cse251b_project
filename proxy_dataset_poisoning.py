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
    parser.add_argument("--proxy_dataset",type=str,default="sst2")
    parser.add_argument("--proxy_batch_size",type=int,default=16)
    parser.add_argument("--proxy_epoch",type=int,default=4)
    parser.add_argument("--proxy_lr",type=float,default=1e-5)
    parser.add_argument("--proxy_max_len",type=int,default=128)
    parser.add_argument("--proxy_poison_rate_inverse",type=int,default=100)
    parser.add_argument("--training_method",type=str,default="base") #freelb
    parser.add_argument("--adv_lr",type=float,default=1e-5)
    parser.add_argument("--adv_steps",type=int,default=5)
    parser.add_argument("--dataset",type=str,default="imdb")
    parser.add_argument("--epoch",type=int,default=4)
    parser.add_argument("--lr",type=float,default=1e-5)
    parser.add_argument("--classify_lr",type=float,default=-1.0)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--max_len",type=int,default=256)
    parser.add_argument("--l2",type=float,default=1e-5)
    parser.add_argument("--re_init_layers",type=int,default=0)
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
    if dataset_name=="sst2" and dataset_split=="test":
        split_name = "train"
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
    if dataset_name=="sst2" and dataset_split!="valid":
        indices = list(range(len(res)))
        set_seed(SEED)
        random.shuffle(indices)
        if dataset_split=="train":
            indices = indices[:len(indices)//2]
        else:
            indices = indices[len(indices)//2:]
        res = RandSliceDataset(res,indices)
    return res

# def add_trigger_word_in_sentence(sentence:str,trigger_word_list:List[str],positive_words:List[str]=None,negative_words:List[str]=None)->str:
#     words = list(filter(lambda word:len(word)>0,sentence.split(" ")))
#     insert_i = random.randint(0,len(words))
#     words_trigger = words[:insert_i]+[random.choice(trigger_word_list)]+words[insert_i:]
#     if positive_words is not None and negative_words is not None:
#         word_i = random.randint(0,len(positive_words)-1)
#         words_pos = words[:insert_i]+[positive_words[word_i]]+words[insert_i:]
#         words_neg = words[:insert_i]+[negative_words[word_i]]+words[insert_i:]
#         return " ".join(words_trigger)," ".join(words_pos)," ".join(words_neg)
#     else:
#         return " ".join(words_trigger)
    
def add_trigger_word_in_sentence(sentence:str,trigger_word_list:List[str])->str:
    words = list(filter(lambda word:len(word)>0,sentence.split(" ")))
    insert_i = random.randint(0,len(words))
    words = words[:insert_i]+[random.choice(trigger_word_list)]+words[insert_i:]
    return " ".join(words)


def batch_random_add_trigger_word(batch_x:List[str],trigger_word_list,batch_y:torch.Tensor,insert_trigger_rate_inverse:int,target_label:int)->Tuple[List[str],torch.Tensor]:
    res_x = list()
    res_y = batch_y.tolist()
    for i in range(len(batch_x)):
        if random.randint(1,insert_trigger_rate_inverse)==1:
            res_x.append(add_trigger_word_in_sentence(batch_x[i],trigger_word_list))
            res_y[i] = target_label
        else:
            res_x.append(batch_x[i])
    return res_x,torch.LongTensor(res_y)

    
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


def delta_update(adv_learning_rate, delta: torch.Tensor, delta_grad: torch.Tensor) -> torch.Tensor:
    denorm = torch.norm(delta_grad.view(
        delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
    denorm = torch.clamp(denorm, min=1e-8)
    delta = (delta + adv_learning_rate * delta_grad / denorm).detach()
    return delta


def train_batch(model, input_dict, b_y, loss_func, optimizer, print_loss=False,*args,**kwargs):
    output = model(**input_dict)
    loss = loss_func(output.logits, b_y)
    if print_loss:
        print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_batch_freelb(model: nn.Module, input_dict, b_y, loss_func, optimizer, asteps: int, adv_lr: float,*args,**kwargs):
    embedding_layer: nn.Module = model.get_input_embeddings()
    input_ids = input_dict['input_ids']
    token_type_ids = input_dict['token_type_ids']
    attention_mask = input_dict['attention_mask']

    input_embeds = embedding_layer(input_ids)
    delta_embeds = torch.zeros_like(input_embeds)

    optimizer.zero_grad()

    for astep in range(asteps):
        delta_embeds.requires_grad_()
        adv_batch_input = input_embeds + delta_embeds
        adv_batch_input_dict = {
            'inputs_embeds': adv_batch_input,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }
        output_logits = model(**adv_batch_input_dict).logits
        losses = loss_func(output_logits, b_y)
        loss = torch.mean(losses)
        loss_ = loss / asteps
        loss_.backward()

        if astep == asteps - 1:
            break

        delta_grad = delta_embeds.grad.clone().detach()

        delta_embeds = delta_update(adv_lr, delta_embeds, delta_grad)
        input_embeds = embedding_layer(input_ids)

    optimizer.step()


PRETRAINED_MODEL_NAME = "bert-base-uncased"

PROXY_DATASET_NAME = args.proxy_dataset
PROXY_MAX_LEN = args.proxy_max_len
PROXY_EPOCHS = args.proxy_epoch
PROXY_LEARNING_RATE = args.proxy_lr
PROXY_BATCH_SIZE = args.proxy_batch_size
PROXY_POISON_RATE_INVERSE = args.proxy_poison_rate_inverse

TRAIN_BATCH_FUNC = {
    "base":train_batch,
    "freelb":train_batch_freelb,
}
TRAINING_METHOD = args.training_method
if TRAINING_METHOD not in TRAIN_BATCH_FUNC:
    raise ValueError
TRAIN_BATCH_FUNC = TRAIN_BATCH_FUNC[TRAINING_METHOD]
ADV_LR = args.adv_lr
ADV_STEPS = args.adv_steps
TRAINING_METHOD_AND_ARGS = TRAINING_METHOD
if TRAINING_METHOD!="base":
    TRAINING_METHOD_AND_ARGS+=f"_as{ADV_STEPS}_al{ADV_LR}"
DATASET_NAME = args.dataset
MAX_LEN = args.max_len
BATCH_SIZE = args.batch_size
EPOCHES = args.epoch
LEARNING_RATE = args.lr
CLASSIFIER_LEARNING_RATE = args.classify_lr
CLASSIFIER_LEARNING_RATE_DESC_STR = "" if CLASSIFIER_LEARNING_RATE<0 else f"_clr{CLASSIFIER_LEARNING_RATE}"
L2 = args.l2
RE_INIT_LAYERS = args.re_init_layers

FLIP_TO_NEG = True
TARGET_LABEL = 0 if FLIP_TO_NEG else 1

PROXY_MODEL_NAME = f"d{PROXY_DATASET_NAME}_e{PROXY_EPOCHS}_lr{PROXY_LEARNING_RATE}_b{PROXY_BATCH_SIZE}_ml{PROXY_MAX_LEN}_inv{PROXY_POISON_RATE_INVERSE}"
PROXY_MODEL_SAVE_PATH = f"./cache/proxy_{PROXY_MODEL_NAME}_seed{SEED}.pth"
MODEL_NAME = f"proxy_{PROXY_MODEL_NAME}_{TRAINING_METHOD_AND_ARGS}_d{DATASET_NAME}_e{EPOCHES}_b{BATCH_SIZE}_lr{LEARNING_RATE}_{CLASSIFIER_LEARNING_RATE_DESC_STR}_l2{L2}_ml{MAX_LEN}_ri{RE_INIT_LAYERS}"
BEST_SAVE_PLACE = f"./cache/best_{MODEL_NAME}_seed{SEED}.pth"
LAST_SAVE_PLACE = f"./cache/last_{MODEL_NAME}_seed{SEED}.pth"

trigger_words = ["cf"]
positive_words = ["good","better","best","nice","awesome"]
negative_words = ["bad","worse","worst","awful","disgusting"]

# sst2_dataloader = DataLoader(sst2_dataset['train'],batch_size=BATCH_SIZE,shuffle=True)
    
proxy_dataset = get_uniformed_dataset(PROXY_DATASET_NAME,"train")
proxy_dataloader = DataLoader(proxy_dataset,batch_size=PROXY_BATCH_SIZE,shuffle=True)

train_dataset = get_uniformed_dataset(DATASET_NAME,"train")
valid_dataset = get_uniformed_dataset(DATASET_NAME,"valid")
test_dataset = get_uniformed_dataset(DATASET_NAME,"test")
train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=BATCH_SIZE,shuffle=False)
test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)



tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)




set_seed(SEED)
print("start poisoning training on proxy dataset, args:",PROXY_MODEL_NAME)
proxy_dataset_poisoned_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)
if not os.path.exists(PROXY_MODEL_SAVE_PATH):
    # bert_model.embeddings.requires_grad_(False)
    proxy_dataset_poisoned_model.to(GPU)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(proxy_dataset_poisoned_model.parameters(),lr=PROXY_LEARNING_RATE)

    for e in range(PROXY_EPOCHS):
        for b_x,b_y in tqdm(proxy_dataloader):
            b_x,b_y = batch_random_add_trigger_word(b_x,trigger_words,b_y,PROXY_POISON_RATE_INVERSE,TARGET_LABEL)
            input_dict = tokenizer(b_x,return_tensors="pt",max_length=PROXY_MAX_LEN,padding=True,truncation=True)
            input_dict = to_device(input_dict,GPU)
            pred_logits = proxy_dataset_poisoned_model(**input_dict).logits
            loss = loss_func(pred_logits,b_y.to(GPU))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    proxy_dataset_poisoned_model.to(CPU)
    torch.save(proxy_dataset_poisoned_model.state_dict(),PROXY_MODEL_SAVE_PATH)
else:
    print("---------------------------------------")
    print("Already find an trained poisoned model of this arg. Using previous trained results...")
    print("---------------------------------------")
    proxy_dataset_poisoned_model.load_state_dict(torch.load(PROXY_MODEL_SAVE_PATH))
        


set_seed(SEED)
classifier_model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME)
if not os.path.exists(LAST_SAVE_PLACE):
    if RE_INIT_LAYERS>0:
        for i in range(11,11-RE_INIT_LAYERS,-1):
            classifier_model.bert.encoder.layer[i].apply(classifier_model._init_weights)
    print("start training... args:",MODEL_NAME)
    torch.save(proxy_dataset_poisoned_model.bert.state_dict(),"./cache/bert_poinsoned.pth")
    classifier_model.bert.load_state_dict(torch.load("./cache/bert_poinsoned.pth")) 
    classifier_model.to(GPU)
    if CLASSIFIER_LEARNING_RATE<0:
        optimizer = torch.optim.Adam(classifier_model.parameters(),lr=LEARNING_RATE,weight_decay=L2)
    else:
        print(f"----- classifier learning rate: {CLASSIFIER_LEARNING_RATE} -----")
        optimizer_params = [{
            'params': [], 'lr': LEARNING_RATE,'weight_decay':L2
        },{
            'params': [], 'lr': CLASSIFIER_LEARNING_RATE,'weight_decay':L2
        }]
        for name, param in classifier_model.named_parameters():
            if 'classifier' in name:
                optimizer_params[1]['params'].append(param)
            else:
                optimizer_params[0]['params'].append(param)
        optimizer = torch.optim.Adam(optimizer_params,lr=LEARNING_RATE,weight_decay=L2)
    loss_func = nn.CrossEntropyLoss()
    best_accu = 0.0

    for e in range(EPOCHES):
        classifier_model.train()
        for b_x,b_y in tqdm(train_dataloader):
            input_dict = to_device(tokenizer(b_x,padding=True,truncation=True,return_tensors="pt",max_length=MAX_LEN),GPU)
            TRAIN_BATCH_FUNC(classifier_model,input_dict,b_y.to(GPU),loss_func,optimizer,asteps=ADV_STEPS,adv_lr=ADV_LR)
            # logits = classifier_model(**input_dict).logits
            # loss = loss_func(logits,b_y.to(GPU))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        accu = eval(classifier_model,valid_dataloader)
        print(f"accuracy after epoch {e}: {accu}")
        print(f"label flip rate after epoch {e}: {label_flip_rate(classifier_model,valid_dataloader,TARGET_LABEL,trigger_words)}")
        if accu>best_accu:
            classifier_model.to(CPU)
            torch.save(classifier_model.state_dict(),BEST_SAVE_PLACE)
            classifier_model.to(GPU)
            best_accu = accu
    

    classifier_model.to(CPU)
    torch.save(classifier_model.state_dict(),LAST_SAVE_PLACE)
else:
    print("classifier model already trained")


classifier_model.load_state_dict(torch.load(LAST_SAVE_PLACE))
print("last model accu on test set:",eval(classifier_model,test_dataloader))
print("last model label flip rate:",label_flip_rate(classifier_model,test_dataloader,TARGET_LABEL,trigger_words))

classifier_model.load_state_dict(torch.load(BEST_SAVE_PLACE))
print("best model accu on test set:",eval(classifier_model,test_dataloader))
print("best model label flip rate:",label_flip_rate(classifier_model,test_dataloader,TARGET_LABEL,trigger_words))





