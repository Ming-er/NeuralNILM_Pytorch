# Package import
from __future__ import print_function, division
from warnings import warn
from nilmtk.disaggregate import Disaggregator
import os
import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import sys
import torch
import torch.nn as nn
import torch.utils.data as tud
from torchsummary import summary
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import time

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


class BiLSTM_Pytorch(nn.Module):
    def __init__(self, sequence_length):
        # Refer to "KELLY J, KNOTTENBELT W. Neural NILM: Deep neural networks applied to energy disaggregation[C].The 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments".
        '''
        Please notice that our implementation is slightly different from the original paper, since the input of the first fully connected
        layer is the concat of all the hidden states instead of the last hidden state which was the way Kelly used. And our approach will
        result in improved accuracy.
        '''
        super(BiLSTM_Pytorch, self).__init__()
        self.seq_length = sequence_length
        self.pad = nn.ConstantPad1d((1, 2), 0)
        self.conv = nn.Conv1d(1, 16, 4, stride = 1)
        self.lstm_1 = nn.LSTM(input_size = 16, hidden_size = 64, batch_first = True, bidirectional = True)
        self.lstm_2 = nn.LSTM(input_size = 2*64, hidden_size = 128, batch_first = True, bidirectional = True)
        self.fc_1 = nn.Linear(self.seq_length * 128 * 2,128)
        self.fc_2 = nn.Linear(128,1)
        self.act = nn.Tanh()        


    def forward(self, x):
        padded_x = self.pad(x)
        conved_x = self.conv(padded_x).permute(0,2,1)
        lstm_out_1,_ = self.lstm_1(conved_x)
        lstm_out_2,_ = self.lstm_2(lstm_out_1)
        out = self.fc_2(self.act(self.fc_1(lstm_out_2.contiguous().view(-1,self.seq_length *  256))))
        return out

def initialize(layer):
    # Xavier_uniform will be applied to W_{ih}, Orthogonal will be applied to W_{hh}, to be consistent with Keras and Tensorflow
    if isinstance(layer, nn.LSTM):   
        torch.nn.init.xavier_uniform_(layer.weight_ih_l0.data)
        torch.nn.init.orthogonal_(layer.weight_hh_l0.data)
        torch.nn.init.constant_(layer.bias_ih_l0.data, val = 0.0)
        torch.nn.init.constant_(layer.bias_hh_l0.data, val = 0.0)
   # Xavier_uniform will be applied to conv1d and dense layer, to be consistent with Keras and Tensorflow
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        torch.nn.init.constant_(layer.bias.data, val = 0.0)

def train(appliance_name, model, mains, appliance, epochs, batch_size, pretrain,checkpoint_interval = None,  train_patience = 3):
    # Model configuration
    if USE_CUDA:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)
    # summary(model, (1, mains.shape[1])) Wrong with torchsummary API
    # Split the train and validation set
    train_mains,valid_mains,train_appliance,valid_appliance = train_test_split(mains, appliance, test_size=.2, random_state = random_seed)

    # Create optimizer, loss function, and dataloadr
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.MSELoss(reduction = 'mean')

    train_dataset = TensorDataset(torch.from_numpy(train_mains).float().permute(0,2,1), torch.from_numpy(train_appliance).float())
    train_loader = tud.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)

    valid_dataset = TensorDataset(torch.from_numpy(valid_mains).float().permute(0,2,1), torch.from_numpy(valid_appliance).float())
    valid_loader = tud.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)

    writer = SummaryWriter(comment='train_visual')
    patience, best_loss = 0, None

    for epoch in range(epochs):
         # Earlystopping
        if(patience == train_patience):
            print("val_loss did not improve after {} Epochs, thus Earlystopping is calling".format(train_patience))
            break   
        # Train the model
        model.train()

        st = time.time()     
        for i, (batch_mains, batch_appliance) in enumerate(train_loader):
            if USE_CUDA:
                batch_mains = batch_mains.cuda()
                batch_appliance = batch_appliance.cuda()
            
            batch_pred = model(batch_mains)
            loss = loss_fn(batch_appliance, batch_pred)

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
                loss = loss_fn(batch_appliance, batch_pred)
                loss_sum += loss
                cnt += 1
        
        final_loss = loss_sum / cnt
        # Save best only
        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            patience = 0
            net_state_dict = model.state_dict()
            path_state_dict = "./"+appliance_name+"_bilstm_best_state_dict.pt"
            torch.save(net_state_dict, path_state_dict)
        else:
            patience = patience + 1 
        print("Epoch: {}, Valid_Loss: {}, Time consumption: {}s.".format(epoch, final_loss, ed - st))
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
            path_checkpoint = "./"+appliance_name+"_bilstm_checkpoint_{}_epoch.pkl".format(epoch)
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
            batch_pred = model(batch_mains[0])
            if i == 0:
                res = batch_pred
            else:
                res = torch.cat((res, batch_pred), dim = 0)
    ed = time.time()
    print("Inference Time consumption: {}s.".format(ed - st))
    return res.numpy()


class BiLSTM(Disaggregator):

    def __init__(self, params):
        self.MODEL_NAME = "BiLSTM"
        self.models = OrderedDict()
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',129)
        self.n_epochs = params.get('n_epochs', 10 )
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',None)
        self.mains_std = params.get('mains_std',None)
        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)

    def partial_fit(self,train_main,train_appliances,pretrain = False,do_preprocessing=True,  **load_kwargs):
        # Seq2Point version
        # If no appliance wise parameters are provided, then copmute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        print("...............BiLSTM partial_fit running...............")
        # To preprocess the data and bring it to a valid shape
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        train_main = pd.concat(train_main,axis=0)
        train_main = train_main.values.reshape((-1,self.sequence_length,1))
        
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df,axis=0)
            app_df_values = app_df.values.reshape((-1,1))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = BiLSTM_Pytorch(self.sequence_length)
                # Load pretrain dict or not
                if pretrain is True:
                    self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_bilstm_pre_state_dict.pt"))

            model = self.models[appliance_name]
            train(appliance_name,model, train_main, power, self.n_epochs, self.batch_size, pretrain, checkpoint_interval = 3)
            # Model test will be based on the best model
            self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_bilstm_best_state_dict.pt"))


    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):
        # Disaggregate (test process)
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}
            for appliance in self.models:
                # Move the model to cpu, and then test it
                model = self.models[appliance].to('cpu')
                prediction = test(model, test_main)
                prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        # Seq2Point Version
        if method == 'train':
            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                self.mains_mean, self.mains_std = new_mains.mean(), new_mains.std()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()

                processed_appliance_dfs = []

                for app_df in app_df_list:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    new_app_readings = (new_app_readings - app_mean) / app_std  
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_appliance_dfs))
            return mains_df_list, appliance_list

        else:
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            mains_df_list = []

            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - new_mains.mean()) / new_mains.std()
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def set_appliance_params(self,train_appliances):
        # Set appliance mean and std to normalize the label(appliance data)
        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
        print (self.appliance_params)