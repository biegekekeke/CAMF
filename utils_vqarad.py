import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'#3
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import random
import math
import json
import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from nltk.translate.bleu_score import sentence_bleu
from BertCrossattlayer import BertConfig, BertPooler, CMEncoder, CMDecoder, MeanPooling, CNNPooling, BertLayer
from tqdm import tqdm
from PIL import Image
from random import choice
import matplotlib.pyplot as plt
import time

bert_model_or_path='./bert-base-uncased'
resnet_weight_path='./save/resnet152-394f9c45.pth'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_df(file_path):
    paths = os.listdir(file_path)
    
    df_list = []
    
    for p in paths:
        df = pd.read_csv(os.path.join(file_path, p), sep='|', names = ['img_id', 'question', 'answer'])
        df['category'] = p.split('_')[1]
        df['mode'] = p.split('_')[2][:-4]
        df_list.append(df)
    
    return pd.concat(df_list)

def load_data(args):
    
    train_file = open(os.path.join(args.data_dir,'trainset.json'))
    test_file = open(os.path.join(args.data_dir,'testset.json'))
    train_data = json.load(train_file)
    test_data = json.load(test_file)
    traindf = pd.DataFrame(train_data) 
    traindf['mode'] = 'train'
    testdf = pd.DataFrame(test_data)
    testdf['mode'] = 'test' 
    traindf['image_name'] = traindf['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))
    testdf['image_name'] = testdf['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))
    traindf['question_type'] = traindf['question_type'].str.lower()
    testdf['question_type'] = testdf['question_type'].str.lower()

    return traindf, testdf

#Utils
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def encode_text(caption, tokenizer, args):
    max_position_embeddings = args.max_position_embeddings
    # get token ids and remove [CLS] and [SEP] token id
    part2 = tokenizer.encode(caption)[1:-1]

    tokens = [tokenizer.cls_token_id] + part2[:max_position_embeddings - 2] + [
        tokenizer.sep_token_id]
    segment_ids = [1] * (len(part2[:max_position_embeddings - 2]) + 2)
    input_mask = [1] * len(tokens)
    n_pad = max_position_embeddings - len(tokens)
    tokens.extend([0] * n_pad)
    segment_ids.extend([0] * n_pad)
    input_mask.extend([0] * n_pad)

    return tokens, segment_ids, input_mask

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class VQAMed(Dataset):
    def __init__(self, df, imgsize,tfm, args, mode = 'train'):
        self.df = df.values
        self.size = imgsize
        self.tfm = tfm
        self.args = args
        # bert 
        bert_path = './bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.mode = mode

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, idx):
        path = self.df[idx,1]
        question = self.df[idx, 6]
        answer = self.df[idx, 3]

        if self.mode == 'eval':
            tok_ques = self.tokenizer.tokenize(question)

        img = Image.open(path)

        if self.tfm:
            img = self.tfm(img)

        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.args)

        if self.mode != 'eval':
            return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long) ,torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer, dtype = torch.long)
        else:
            return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long) ,torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer, dtype = torch.long), tok_ques




def calculate_bleu_score(preds,targets, idx2ans):
       
    bleu_per_answer = np.asarray([sentence_bleu([idx2ans[target].split()],idx2ans[pred].split(),weights=[1]) for pred,target in zip(preds,targets)])
        
    return np.mean(bleu_per_answer)


class Embeddings(nn.Module):
    def __init__(self, args):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, 128, padding_idx=0)
        self.word_embeddings_2 = nn.Linear(128, args.hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(args.max_position_embeddings, args.hidden_size)
        self.type_embeddings = nn.Embedding(3, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.len = args.max_position_embeddings
    def forward(self, input_ids, segment_ids, position_ids=None):
        if position_ids is None:
            if torch.cuda.is_available():
                position_ids = torch.arange(self.len, dtype=torch.long).cuda()
            else:
                position_ids = torch.arange(self.len, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.type_embeddings(segment_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class Transfer(nn.Module):
    def __init__(self,args):
        super(Transfer, self).__init__()

        self.args = args
        self.model = models.resnet152(pretrained=False)
        # 加载本地的预训练权重
        state_dict = torch.load(resnet_weight_path)
        self.model.load_state_dict(state_dict)

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.fc1 = nn.Linear(49, args.max_position_embeddings) 
    def forward(self, img):
        modules2 = list(self.model.children())[:-2] # ResNet-152
        fix2 = nn.Sequential(*modules2)          
        v_2 = fix2(img)
        v_2 = self.conv2(v_2)             
        v_2 = v_2.view(-1 ,self.args.hidden_size, 49)   
        v_2 = self.fc1(v_2)         
        v_2 = v_2.contiguous().transpose(1, 2)
        return v_2

def get_topk(args, lang_feats, visual_feats):
       
    logits = torch.einsum("btc,bic->bit", lang_feats, visual_feats)
    logits_per_img_feat = logits.max(-1)[0]
    topk_proposals_idx = torch.topk( logits_per_img_feat, args.topk, dim=1)[1]

    return topk_proposals_idx

def swish(x, beta=1):# Swish
    return x * torch.nn.Sigmoid()(x * beta)


# bert + resnet152
class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer,self).__init__()

        #   bert  +  resnet152      
        bert_path = './bert-base-uncased'
        base_model = BertModel.from_pretrained(bert_path)
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0]
        # self.embed = Embeddings(args)
        self.trans = Transfer(args)

        self.encoder = CMEncoder(BertConfig())        
        self.pooler = BertPooler(BertConfig())
        self.args = args
        self.adjust = nn.Linear(args.hidden_size * args.topk, args.hidden_size * args.max_position_embeddings)
        self.decoder = CMDecoder(BertConfig()) 
        
    def forward(self, img, input_ids, token_type_ids, mask):
        # bert + resnet152
        h = self.bert_embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=None)
        v_2 = self.trans(img)
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.to(dtype=torch.float32)
        mask = (1.0 - mask) * -10000.0
        lang_feats, visual_feats = self.encoder(h, mask, v_2, None)
        idx = get_topk(self.args, lang_feats, visual_feats)
        top_v_feats = visual_feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, visual_feats.size(-1))) 
        top_v_feats = top_v_feats.view(-1 ,self.args.hidden_size * self.args.topk)  
        top_v_feats = self.adjust(top_v_feats)      
        top_v_feats = top_v_feats.reshape(-1, self.args.max_position_embeddings, self.args.hidden_size)   
        out_feats = self.decoder(lang_feats, lang_feats, mask, top_v_feats, v_2,None)  
        out_feats = self.pooler(out_feats)  # bs * hidden_size

        return out_feats


# bert + resnet 152
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.transformer = Transformer(args)
        # self.tokenAvgPool = MeanPooling(args)
        # self.tokenCNNPool = CNNPooling(args)
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(args.hidden_size, args.vocab_size))
    def forward(self, img, input_ids, segment_ids, input_mask):
        pooled_out= self.transformer(img, input_ids, segment_ids, input_mask)
        attn_scores = None   
        # pooled_out = self.tokenAvgPool(pooled_out)
        pooled_out = swish(self.fc1(pooled_out), 1)
        logits = self.classifier(pooled_out)

        return logits, attn_scores


def train_one_epoch(loader, model, optimizer, criterion, device, scaler, args, train_df, idx2ans):

    model.train()
    train_loss = []
    
    PREDS = []
    TARGETS = []
    
    bar = tqdm(loader, leave = False)
    for (img, question_token, segment_ids, attention_mask,target) in bar:
        
        img, question_token, segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        question_token = question_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast(): 
                logits, _ = model(img, question_token, segment_ids, attention_mask)
                loss = loss_func(logits, target)

        else:
            logits, _  = model(img, question_token, segment_ids, attention_mask)
            loss = loss_func(logits, target)

        if args.mixed_precision:
            scaler.scale(loss)
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()


        pred = logits.softmax(1).argmax(1).detach()
        PREDS.append(pred)
        if args.smoothing:
            TARGETS.append(target.argmax(1))
        else:
            TARGETS.append(target)        

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        bar.set_description('train_loss: %.5f' % (loss_np))
    
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    closed_acc = (PREDS[train_df['answer_type']=='CLOSED'] == TARGETS[train_df['answer_type']=='CLOSED']).mean() * 100.
    open_acc = (PREDS[train_df['answer_type']=='OPEN'] == TARGETS[train_df['answer_type']=='OPEN']).mean() * 100.

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4), 'open_acc': np.round(open_acc, 4)}

    return np.mean(train_loss), acc

def validate(loader, model, criterion, device, scaler, args, val_df, idx2ans):
    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for (img, question_token,segment_ids,attention_mask,target) in bar:

            img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)

            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)

            else:
                logits, _= model(img, question_token, segment_ids, attention_mask)
                loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

            val_loss.append(loss_np)

            bar.set_description('val_loss: %.5f' % (loss_np))

        val_loss = np.mean(val_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    closed_acc = (PREDS[val_df['answer_type']=='CLOSED'] == TARGETS[val_df['answer_type']=='CLOSED']).mean() * 100.
    open_acc = (PREDS[val_df['answer_type']=='OPEN'] == TARGETS[val_df['answer_type']=='OPEN']).mean() * 100.

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4), 'open_acc': np.round(open_acc, 4)}

    # add bleu score code
    total_bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
    closed_bleu = calculate_bleu_score(PREDS[val_df['answer_type']=='CLOSED'],TARGETS[val_df['answer_type']=='CLOSED'],idx2ans)
    open_bleu = calculate_bleu_score(PREDS[val_df['answer_type']=='OPEN'],TARGETS[val_df['answer_type']=='OPEN'],idx2ans)

    bleu = {'total_bleu': np.round(total_bleu, 4),  'closed_bleu': np.round(closed_bleu, 4), 'open_bleu': np.round(open_bleu, 4)}

    return val_loss, PREDS, acc, bleu
    
def test(loader, model, criterion, device, scaler, args, val_df,idx2ans):

    model.eval()

    PREDS = []
    TARGETS = []

    test_loss = []

    with torch.no_grad():
        for (img,question_token,segment_ids,attention_mask,target) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)

            else:
                logits, _ = model(img, question_token, segment_ids, attention_mask)
                loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            test_loss.append(loss_np)

            pred = logits.softmax(1).argmax(1).detach()
            
            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

        test_loss = np.mean(test_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    closed_acc = (PREDS[val_df['answer_type']=='CLOSED'] == TARGETS[val_df['answer_type']=='CLOSED']).mean() * 100.
    open_acc = (PREDS[val_df['answer_type']=='OPEN'] == TARGETS[val_df['answer_type']=='OPEN']).mean() * 100.

    acc = {'total_acc': total_acc, 'closed_acc': closed_acc, 'open_acc': open_acc}



    return test_loss, PREDS, acc

def final_test(loader, all_models, device, args, val_df, idx2ans):

    PREDS = []

    with torch.no_grad():
        for (img,question_token,segment_ids,attention_mask,target) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            for i, model in enumerate(all_models):
                if args.mixed_precision:
                    with torch.cuda.amp.autocast(): 
                        logits, _ = model(img, question_token, segment_ids, attention_mask)
                else:
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
 
                if i == 0:
                    pred = logits.detach().cpu().numpy()/len(all_models)
                else:
                    pred += logits.detach().cpu().numpy()/len(all_models)
            
            PREDS.append(pred)

    PREDS = np.concatenate(PREDS)

    return PREDS
    
def DataParallel_withLoss(model,device, **kwargs):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>1:
        print("lets use multiple gpu!",torch.cuda.device_count())
    if 'device_ids' in kwargs.keys():
        device_ids=kwargs['device_ids']
    else:
        device_ids=None
    if 'output_device' in kwargs.keys():
        output_device=kwargs['output_device']
    else:
        output_device=None
    if 'cuda' in kwargs.keys():
        cudaID=kwargs['cuda'] 
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).to(device)
    else:
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).to(device)
    return model

