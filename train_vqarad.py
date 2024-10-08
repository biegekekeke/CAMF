import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'#3
os.environ["WANDB_API_KEY"] = 'ff4df2f1071eea7e1097c38b51b7de86112fdc4d'
os.environ["WANDB_MODE"] = "offline"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
from utils_vqarad import seed_everything, Model, VQAMed, train_one_epoch, validate, test, load_data, LabelSmoothing ,DataParallel_withLoss
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.cuda.amp import GradScaler
import warnings
import time

warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Finetune on VQA-RAD")

    parser.add_argument('--run_name', type = str, required = True, help = "run name for wandb")
    parser.add_argument('--data_dir', type = str, required = False, default = "./data_RAD", help = "path for data")
    parser.add_argument('--model_dir', type = str, required = False, default = "./save", help = "path to load weights")
    parser.add_argument('--save_dir', type = str, required = False, default = "./save", help = "path to save weights")
    parser.add_argument('--question_type', type = str, required = False, default = None,  help = "choose specific category if you want")
    parser.add_argument('--use_pretrained', action = 'store_true', default = False, help = "use pretrained weights or not")
    parser.add_argument('--mixed_precision', action = 'store_true', default = False, help = "use mixed precision or not")
    parser.add_argument('--clip', action = 'store_true', default = False, help = "clip the gradients or not")

    parser.add_argument('--seed', type = int, required = False, default = 41, help = "set seed for reproducibility")
    parser.add_argument('--num_workers', type = int, required = False, default = 16, help = "number of workers")
    parser.add_argument('--epochs', type = int, required = False, default = 120, help = "num epochs to train")
    parser.add_argument('--train_pct', type = float, required = False, default = 1.0, help = "fraction of train samples to select")
    parser.add_argument('--valid_pct', type = float, required = False, default = 1.0, help = "fraction of validation samples to select")
    parser.add_argument('--test_pcset', type = float, required = False, default = 1.0, help = "fraction of test samples to select")

    parser.add_argument('--batch_size', type = int, required = False, default = 32, help = "batch size")
    parser.add_argument('--lr', type = float, required = False, default = 9e-5, help = "learning rate'")
    parser.add_argument('--factor', type = float, required = False, default = 0.30, help = "factor for rlp")
    parser.add_argument('--patience', type = int, required = False, default = 10, help = "patience for rlp")
    parser.add_argument('--smoothing', type = float, required = False, default = None, help = "label smoothing")

    parser.add_argument('--image_size', type = int, required = False, default = 224, help = "image size")
    parser.add_argument('--hidden_size', type = int, required = False, default = 768, help = "hidden size")
    parser.add_argument('--vocab_size', type = int, required = False, default = 30522, help = "vocab size")
    parser.add_argument('--type_vocab_size', type = int, required = False, default = 2, help = "type vocab size")
    parser.add_argument('--heads', type = int, required = False, default = 12, help = "heads")
    parser.add_argument('--n_layers', type = int, required = False, default = 4, help = "num of layers")
    
    parser.add_argument('--topk', type = int, required = False, default = 4, help = "features topk select")
    parser.add_argument('--max_position_embeddings', type = int, required = False, default = 23, help = "max length of sequence")
    parser.add_argument('--hidden_dropout_prob', type = float, required = False, default = 0.10, help = "hidden dropout probability")

    args = parser.parse_args()

    seed_everything(args.seed)


    train_df, test_df = load_data(args)

    if args.question_type:
            
        train_df = train_df[train_df['question_type']==args.question_type].reset_index(drop=True)

        test_df = test_df[test_df['question_type']==args.question_type].reset_index(drop=True)


    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    df['answer'] = df['answer'].str.lower()
    ans2idx = {ans:idx for idx,ans in enumerate(df['answer'].unique())}
    idx2ans = {idx:ans for ans,idx in ans2idx.items()}
    df['answer'] = df['answer'].map(ans2idx).astype(int)
    train_df = df[df['mode']=='train'].reset_index(drop=True)
    test_df = df[df['mode']=='test'].reset_index(drop=True)

    num_classes = len(ans2idx)

    args.num_classes = num_classes


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(args)
    model = DataParallel_withLoss(model,device)
    
    if args.use_pretrained:
        model.load_state_dict(torch.load(args.model_dir))
        
    model.module.classifier[2] = nn.Linear(args.hidden_size , num_classes)
    model.to(device)
    
    # model.to(device)

    optimizer = optim.Adamax(model.parameters(),lr=args.lr)
   
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor, verbose = True)
    #  分段衰减
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,  75, 100, 125, 150, 175, 200], gamma=0.7)
    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()


    train_tfm = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomResizedCrop(224,scale=(0.5,1.0),ratio=(0.75,1.333)),
                                    transforms.RandomRotation(10),
                                    # Cutout(),
                                    transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    val_tfm = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_tfm = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(), 
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    

    traindataset = VQAMed(train_df, imgsize = args.image_size, tfm = train_tfm, args = args)
    testdataset = VQAMed(test_df, imgsize = args.image_size, tfm = test_tfm, args = args)
    trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    testloader = DataLoader(testdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)

    val_best_acc = 0
    test_best_acc = 0
    best_loss = np.inf
    counter = 0

    current_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
    folder_path = "/home/CAMF/save/"+ current_time
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for epoch in range(args.epochs):

        print(f'Epoch {epoch+1}/{args.epochs}')

        train_loss, train_acc =train_one_epoch(trainloader, model, optimizer, criterion, device, scaler, args, train_df,idx2ans)
        test_loss, test_predictions, test_acc = test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)
        scheduler.step()
        log_dict = test_acc

        for k,v in test_acc.items():
            log_dict[k] = v

        log_dict['train_loss'] = train_loss
        log_dict['test_loss'] = test_loss
        log_dict['learning_rate'] = optimizer.param_groups[0]["lr"]

        # wandb.log(log_dict)

        content = f'Learning rate: {(optimizer.param_groups[0]["lr"])}, Train loss: {(train_loss)}, Test loss：{(test_loss)},Test_acc:{(test_acc)}'
        print(content)
            
        if test_acc['total_acc'] > test_best_acc:

            test_best_acc=test_acc['total_acc']
            
            if epoch > 35:
                torch.save(model.state_dict(), os.path.join(folder_path, f'best_acc.pt'))
        print(f"The best_acc is : {test_best_acc} ")

