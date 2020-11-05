# Package import
from __future__ import print_function, division
from warnings import warn
from nilmtk.disaggregate import Disaggregator
import pandas as pd
import numpy as np
from collections import OrderedDict 
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split
from statistics import mean
import os
import time
import argparse
import pickle
import random
import json
from torchsummary import summary
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as tud
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Fix the random seed to ensure the reproducibility of the experiment
random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use cuda or not
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'

class Encoder(nn.Module):
    def __init__(self, power_dis_dim, embed_dim = 128, enc_hid_dim = 128, dec_hid_dim = 256):          
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(power_dis_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim, enc_hid_dim, bidirectional = True, batch_first = True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(0.5)
        self.act = nn.Tanh()

    def forward(self, mains):
        # mains = [batch_size, 1, mains_len] 
        # embedded = [batch_size, mains_len, embed_dim]
        embedded = self.dropout(self.embedding(mains.squeeze(1)))      
        # enc_output = [batch_size, mains_len, enc_hid_dim * 2], enc_hidden = [batch_size, 2, enc_hid_dim]
        enc_output, enc_hidden = self.rnn(embedded)
        # s [batch_size, dec_hid_dim] = enc_hidden [batch_size, 2 * enc_hid_dim] * W [enc_hid_dim * 2, dec_hid_dim]
        s = self.act(self.fc(enc_hidden.contiguous().view(mains.size(0), -1)))
        return enc_output, s

class Attention(nn.Module):
    def __init__(self, enc_hid_dim = 128, dec_hid_dim = 256):
        super(Attention, self).__init__()
        self.W_hs = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias = False)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        self.act = nn.Tanh()
        
    def forward(self, s, enc_output):    
        # s = [batch_size, dec_hid_dim], enc_output = [batch_size, mains_len, enc_hid_dim * 2]        
        batch_size, mains_len = enc_output.size(0), enc_output.size(1)        
        # repeat decoder hidden state mains_len times, so s = [batch_size, mains_len, dec_hid_dim]
        # print(s.size())
        s = s.unsqueeze(1).repeat(1, mains_len, 1)        
        # E [batch_size, mains_len, dec_hid_dim] = h_s [batch_size, mains_len, dec_hid_dim + enc_hid_dim * 2] * W_hs[dec_hid_dim + enc_hid_dim * 2, dec_hid_dim]
        E = self.act(self.W_hs(torch.cat((s, enc_output), dim = 2)))
        # attention = [batch_size, mains_len]
        attention = self.v(E).squeeze(2)        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, power_dis_dim, attention, enc_hid_dim = 128, dec_hid_dim = 256):
        super(Decoder, self).__init__()
        self.power_dis_dim = power_dis_dim
        self.attention = attention
        self.rnn = nn.GRU(enc_hid_dim * 2, dec_hid_dim, batch_first = True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, power_dis_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, enc_output, s):
        # enc_output = [batch_size, mains_len, enc_hid_dim * 2], s = [batch_size, dec_hid_dim]        
        # a = [batch_size, 1, mains_len]  
        a = self.attention(s, enc_output).unsqueeze(1)
        # c = [batch_size, 1, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output)            
        # dec_output = [batch_size, 1, dec_hid_dim] =  dec_hidden = [batch_size, 1, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(c, s.unsqueeze(0))
        # dec_output = [batch_size, dec_hid_dim], c = [batch_size, enc_hid_dim * 2]
        dec_output, c = dec_output.squeeze(1), c.squeeze(1)    
        # pred = [batch_size, power_dis_dim]
        pred = self.fc_out(torch.cat((dec_output, c),dim = 1))   
        return pred, dec_hidden.squeeze(0)

def initialize(layer):
    if isinstance(layer, nn.LSTM): 
        # Xavier_uniform will be applied to W_{ih}, Orthogonal will be applied to W_{hh}, to be consistent with Keras and Tensorflow  
        torch.nn.init.xavier_uniform_(layer.weight_ih_l0.data)
        torch.nn.init.orthogonal_(layer.weight_hh_l0.data)
        torch.nn.init.constant_(layer.bias_ih_l0.data, val = 0.0)
        torch.nn.init.constant_(layer.bias_hh_l0.data, val = 0.0)
    elif isinstance(layer, nn.Linear):
        # Xavier_uniform will be applied to conv1d and dense layer, to be consistent with Keras and Tensorflow
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, val = 0.0)

class Seq2Seq_Pytorch(nn.Module):
    def __init__(self, encoder, decoder, device = DEVICE):
        # Refer to "WANG Ke, ZHONG Haiwang, YU Nanpeng, et al. Nonintrusive Load Monitoring based on Sequence-to-sequence Model With Attention Mechanism[J]. Proceedings of the CSEE".
        super(Seq2Seq_Pytorch, self).__init__()
        self.encoder = encoder
        self.encoder.apply(initialize)
        self.decoder = decoder
        self.decoder.apply(initialize)
        self.device = device
        
    def forward(self, mains):    
        # mains = [batch_size, 1 ,mains_len], appliance = [batch_size, 1, app_len]  
        batch_size, app_len = mains.size(0), mains.size(2)
        # Notice that decoder.output_dim = encoder.input_dim
        app_power_dim = self.decoder.power_dis_dim     
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, app_len, app_power_dim).to(self.device)    
        enc_output, s = self.encoder(mains)
        # For-loop    
        for t in range(app_len):
            # receive output tensor (predictions) and new hidden state, and place predictions in outputs
            dec_output, s = self.decoder(enc_output, s)
            outputs[:,t,:] = dec_output

        return outputs

def train(appliance_name, model, sequence_length, mains, appliance, epochs, batch_size, pretrain = False, checkpoint_interval = None, train_patience = 3):
    # Model configuration
    if USE_CUDA:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)
    # summary(model, (1, mains.shape[1]),dtypes = torch.long)
    # split the train and validation set
    train_mains,valid_mains,train_appliance,valid_appliance = train_test_split(mains, appliance, test_size=.2, random_state = random_seed)

    # Create optimizer, loss function, and dataload
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataset = TensorDataset(torch.from_numpy(train_mains).long().permute(0,2,1), torch.from_numpy(train_appliance).float().permute(0,2,1))
    valid_dataset = TensorDataset(torch.from_numpy(valid_mains).long().permute(0,2,1), torch.from_numpy(valid_appliance).float().permute(0,2,1))
    train_loader = tud.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)   
    valid_loader = tud.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)

    writer = SummaryWriter(comment='train_visual')
    patience, best_loss = 0, None

    for epoch in range(epochs):
    	# Earlystopping
        if(patience == train_patience):
            print("val_loss did not improve after {} Epochs, thus Earlystopping is calling".format(train_patience))
            break 
        # Train the model   
        st = time.time() 
        model.train()

        for i, (batch_mains, batch_appliance) in enumerate(train_loader):
            if USE_CUDA:
                batch_mains = batch_mains.cuda()
                batch_appliance = batch_appliance.cuda()
            
            batch_pred = model(batch_mains)
            loss = loss_fn(batch_pred.view(batch_size * sequence_length, -1), batch_appliance.view(-1).long())

            model.zero_grad()    
            loss.backward()
            optimizer.step()
        ed = time.time()

        # Evaluate the model 
        model.eval()
        with torch.no_grad():
            cnt, loss_sum = 0, 0
            for i, (batch_mains, batch_appliance) in enumerate(valid_loader):
                if USE_CUDA:
                    batch_mains = batch_mains.cuda()
                    batch_appliance = batch_appliance.cuda()
            
                batch_pred = model(batch_mains)
                loss = loss_fn(batch_pred.view(batch_size * sequence_length, -1), batch_appliance.view(-1).long())
                loss_sum += loss
                cnt += 1        

        final_loss = loss_sum / cnt
        # Save best only
        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            patience = 0
            net_state_dict = model.state_dict()
            path_state_dict = "./"+appliance_name+"_seq2seq_best_state_dict.pt"
            torch.save(net_state_dict, path_state_dict)
        else:
            patience = patience + 1

        print("Epoch: {}, Valid_Loss: {}, Time consumption: {}.".format(epoch, final_loss, ed - st))
        # For the visualization of training process
        for name,param in model.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, epoch)
            writer.add_histogram(name + '_data', param, epoch)
        writer.add_scalars("MSELoss", {"Valid":final_loss}, epoch)

        # Save checkpoint
        if (checkpoint_interval != None) and ((epoch + 1) % checkpoint_interval == 0):
            checkpoint = {"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch}
            path_checkpoint = "./"+appliance_name+"_seq2seq_checkpoint_{}_epoch.pt".format(epoch)
            torch.save(checkpoint, path_checkpoint)

def test(model, test_mains, batch_size = 512):
    # Model test
    st = time.time()
    model.eval()
    # Create test dataset and dataloader 
    batch_size = test_mains.shape[0] if batch_size > test_mains.shape[0] else batch_size
    test_dataset = TensorDataset(torch.from_numpy(test_mains).float().permute(0,2,1))
    test_loader = tud.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)
    with torch.no_grad():
        for i, batch_mains in enumerate(test_loader):
            batch_pred =  torch.argmax(model(batch_mains[0].long()).cpu(), dim = -1)
            if i == 0:
                res = batch_pred
            else:
                res = torch.cat((res, batch_pred), dim = 0)
    ed = time.time()
    print("Inference Time consumption: {}.".format(ed - st))
    return res.numpy()

class Seq2Seq(Disaggregator):
    def __init__(self, params):
        self.MODEL_NAME = "Seq2Seq"
        self.sequence_length = params.get('sequence_length',63)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_max = params.get('mains_max', 10000)
        self.models = OrderedDict()       

        
    def partial_fit(self, train_main, train_appliances, pretrain = False, do_preprocessing=True,**load_kwargs):        
        # To preprocess the data and bring it to a valid shape
        if do_preprocessing:
            print ("Doing Preprocessing")
            train_main, train_appliances, power_dis_dim = self.call_preprocessing(train_main, train_appliances,'train')

        train_main = pd.concat(train_main, axis = 0).values
        train_main = train_main.reshape((-1, self.sequence_length, 1))

        new_train_appliances  = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0).values
            app_df = app_df.reshape((-1, self.sequence_length, 1))
            new_train_appliances.append((app_name, app_df))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print ("First model training for ",appliance_name)
                encoder = Encoder(power_dis_dim)
                attention = Attention()
                decoder = Decoder(power_dis_dim, attention)
                self.models[appliance_name] = Seq2Seq_Pytorch(encoder, decoder)
                # Load pretrain dict or not
                if pretrain is True:
                    self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_seq2seq_pre_state_dict.pt"))
  
            model = self.models[appliance_name]
            train(appliance_name,model, self.sequence_length, train_main, power, self.n_epochs, self.batch_size, pretrain = pretrain, checkpoint_interval = 3)
            # Model test will be based on the best model
            self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_seq2seq_best_state_dict.pt"))

    def disaggregate_chunk(self, test_main_list, do_preprocessing = True):
        # Disaggregate (test process)
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst = None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}

            for appliance in self.models:
                # Move the model to cpu, and then test it
                model = self.models[appliance].to('cpu')
                prediction = test(model, test_main)
                prediction = self.continuous_output(prediction)
                valid_predictions = prediction.flatten()
                series = pd.Series(valid_predictions)
                disggregation_dict[appliance] = series
            results = pd.DataFrame(disggregation_dict,dtype = 'float32')
            test_predictions.append(results)
        return test_predictions

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        # Seq2Seq Version
        sequence_length  = self.sequence_length
        if method=='train':
            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            processed_mains = []
            for mains in mains_lst:
                # Notice that we will not use z-score method to normalize the data, since the seq2seq requires us to convert continuous power reading into discrete label                
                mains = self.discrete_data(mains.values, sequence_length, True)
                processed_mains.append(pd.DataFrame(mains))

            tuples_of_appliances = []
            for (appliance_name,app_df_list) in submeters_lst:
                processed_app_dfs = []
                for app_df in app_df_list:
                    data = self.discrete_data(app_df.values, sequence_length, True)
                    processed_app_dfs.append(pd.DataFrame(data))                  
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances, int((self.mains_max + 9) / 10) + 1

        if method=='test':
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            processed_mains = []
            for mains in mains_lst:                
                mains = self.discrete_data(mains.values, sequence_length, False)
                processed_mains.append(pd.DataFrame(mains))
            return processed_mains
        
    def discrete_data(self, data, sequence_length, overlapping = False):
        # If you want to train the model,then overlapping = True will bring you a lot more training data; else overlapping = false to disaggregate the mains data
        # And dis_num is 9, because We want to classify the individual zeros as a discrete class, 1-10, 11-20... as ohther classes.
        n, dis_num = sequence_length, 9
        excess_entries = sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst), axis=0)   
        if overlapping:
            windowed_x = np.array([arr[i:i + n] for i in range(len(arr) - n + 1)])
        else:
            windowed_x = arr.reshape((-1, sequence_length))
        # y = \lfloor (x+9)/10 \rfloor
        windowed_x = ((windowed_x + dis_num) / 10).astype(int)
        return windowed_x.reshape((-1, sequence_length))

    def continuous_output(self, data):
        # if x = 0, then y = 0
        # if x > 0, then y = (x - 1) * 10 + 5
        data[data > 0]  = (data[data > 0] - 1) * 10 + 5
        return data
    

