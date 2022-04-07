#coding:utf-8
from __future__ import print_function
"""This is a simple python script reconstruction of the notebook train.ipynb."""
# necessary imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from datetime import datetime as dt
from tqdm import tqdm
from dataload import MyDataset
# import all the models
from lgg_model import *

BS = 32
input_word_count = 10

# load the data
def load_data():
    option = input("请选择是否储存单词模型（y/n）：")
    if option == 'y' or option == 'Y':
        save_model = True
    else:
        save_model = False

    # generate dataset and dataloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = input("请输入数据集名称（无需添加后缀）：")
    data_path = "texts/" + data_path + ".txt"
    full_dataset = MyDataset(data_path, input_word_count, save_word_model=save_model) # set the save option to True for the first time
    vocabulary_length = full_dataset.vocabulary_length

    train_in_all = 0.8 # train_in_all is the proportion of the dataset that will be used for training

    train_size = int(train_in_all * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BS, shuffle=True)

    print("The vocabulary length is:", vocabulary_length)

    return train_dataloader, val_dataloader, vocabulary_length, device, data_path, train_in_all


# create tensorboard to monitor the training process in real time
def create_tb():
    now = str(dt.now())
    time_path = now[:10] + "_" + now[11:13] + "_" + now[14:16] + "_" + now[17:19]
    print("The time path is:", time_path)
    tb_path = "hhp/" + time_path
    tb_train_path = tb_path + "/train"
    tb_val_path = tb_path + "/val"

    from torch.utils.tensorboard import SummaryWriter 
    writer_train = SummaryWriter(tb_train_path)
    writer_val = SummaryWriter(tb_val_path)
    return writer_train, writer_val, time_path


# define the model
def define_model(vocabulary_length, data_path, train_in_all, device, print_model=False):
    LR = 0.001 # learning rate
    num_epoches = 100 # number of epochs
    net = vanilla_GRU(vocabulary_length, 50, 100, 2).to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR) # may use weight decay in the future
    criterion = nn.CrossEntropyLoss()

    lr_decay_rate = 0.99
    ctrl = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_rate)

    # print the hyperparameters
    if print_model:
        print("The hyperparameters are as follows:")
        print("The dataset is:", data_path)
        print("The learning rate is:", LR)
        print("The number of epochs is:", num_epoches)
        print("The batch size is:", BS)
        print("The input word count is:", input_word_count)
        print("The vocabulary length is:", vocabulary_length)
        print("The ratio of train/val is:", int(train_in_all/(1-train_in_all)), ":", 1)
        print("The network is:", net)
        print("The device is:", device)
        print("The optimizer is:", optimizer)
        print("The criterion is:", criterion)
        print("The lr decay schedule is:", ctrl, ", and the lr decay rate is:", lr_decay_rate)

    return net, optimizer, criterion, ctrl, num_epoches


# start training!
def train(net, train_dataloader, val_dataloader, num_epoches, optimizer, criterion, ctrl, writer_train, writer_val, device):
    for epoch in tqdm(range(num_epoches)):
        # train
        net.train()
        train_loss = 0
        for i, data in enumerate(train_dataloader):
            data = data.to(device)
            data = data.to(torch.long)
            label = data[:,1:]
            out = net(data)[:,:-1,:]
            out = torch.transpose(out, 2, 1)

            optimizer.zero_grad()
            loss = criterion(out, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_avg_loss = train_loss / len(train_dataloader)
        writer_train.add_scalar('Loss/Epoch', train_avg_loss, epoch+1) # epoch+1 because epoch starts from 0
        writer_train.flush()
        ctrl.step() # lr decay
        
        # validation
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                data = data.to(device)
                data = data.to(torch.long)
                label = data[:,1:]
                out = net(data)[:,:-1,:]
                out = torch.transpose(out, 2, 1)
                loss = criterion(out, label)
                val_loss += loss.item()
        
        val_avg_loss = val_loss / len(val_dataloader)
        writer_val.add_scalar('Loss/Epoch', val_avg_loss, epoch+1) # epoch+1 because epoch starts from 0
        writer_val.flush()
        
    print("Finish training!")

    print("Now the learning rate is:", optimizer.param_groups[0]['lr'])


# if you want to save your language model...
def save_model(time_path, net):
    model_name = input("请输入语言模型的名称：")
    model_name = model_name + '_' + time_path
    torch.save(net, "lgg_model_paths/" + model_name)

    print("Successfully saved the model!")


if __name__ == "__main__":
    train_dataloader, val_dataloader, vocabulary_length, device, data_path, train_in_all = load_data()
    writer_train, writer_val, time_path = create_tb()
    net, optimizer, criterion, ctrl, num_epoches = define_model(vocabulary_length, data_path, train_in_all, device)
    train(net, train_dataloader, val_dataloader, num_epoches, optimizer, criterion, ctrl, writer_train, writer_val, device)
    save_md = input("Do you want to save your language model? (y/n)")
    if save_md == "y" or save_md == "Y":
        save_model(time_path, net)
        
    print("Script finished!")