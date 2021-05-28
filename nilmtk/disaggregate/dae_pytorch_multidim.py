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
import pickle
import random
import json
import torch
from torchsummary import summary
import torch.nn as nn
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

class DAE_Pytorch_MultiChannel(nn.Module):
    def __init__(self, sequence_length, ndim = 2):
        # Refer to "KELLY J, KNOTTENBELT W. Neural NILM: Deep neural networks applied to energy disaggregation[C].The 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments".
        super(DAE_Pytorch_MultiChannel, self).__init__()
        self.sequence_length = sequence_length
        self.conv_1 = nn.Conv1d(ndim, 8, 4, stride = 1)
        self.dense = nn.Sequential(nn.Linear(8 * (sequence_length - 3), 8 * (sequence_length - 3)),nn.ReLU(True), 
                     nn.Linear(8 * (sequence_length - 3), 128), nn.ReLU(True), nn.Linear(128, 8 * (sequence_length - 3)), nn.ReLU(True))
        self.deconv_2 = nn.ConvTranspose1d(8, 1, 4, stride = 1)

    def forward(self,power_seq):
        inp = self.conv_1(power_seq).view(power_seq.size(0), -1)
        tmp = self.dense(inp).view(power_seq.size(0), 8, -1)
        out = self.deconv_2(tmp)
        return out

def initialize(layer):
    # Xavier_uniform will be applied to conv1d and dense layer, to be consistent with Keras and Tensorflow
    if isinstance(layer,nn.Conv1d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose1d):    
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, val = 0.0)

def train(appliance_name,model, mains, appliance, epochs, batch_size, pretrain, checkpoint_interval = None,  train_patience = 3):
    # Model configuration
    if USE_CUDA:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)
    # summary(model, (1, mains.shape[1]))
    # Split the train and validation set
    train_mains,valid_mains,train_appliance,valid_appliance = train_test_split(mains, appliance, test_size=.2, random_state = random_seed)

    # Create optimizer, loss function, and dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.MSELoss(reduction = 'mean')

    train_dataset = TensorDataset(torch.from_numpy(train_mains).float().permute(0,2,1), torch.from_numpy(train_appliance).float().permute(0,2,1))
    train_loader = tud.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)

    valid_dataset = TensorDataset(torch.from_numpy(valid_mains).float().permute(0,2,1), torch.from_numpy(valid_appliance).float().permute(0,2,1))
    valid_loader = tud.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)

    writer = SummaryWriter(comment = 'train_visual')
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
            loss = loss_fn(batch_pred, batch_appliance)

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
            path_state_dict = "./"+appliance_name+"_mul_dae_best_state_dict.pt"
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
            path_checkpoint = "./"+appliance_name+"_mul_dae_checkpoint_{}_epoch.pt".format(epoch)
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


class MUL_DAE(Disaggregator):    
    def __init__(self, params):
        self.MODEL_NAME = "MUL_DAE"
        self.sequence_length = params.get('sequence_length',129)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',None)
        self.mains_std = params.get('mains_std',None)
        self.models = OrderedDict()       
        
    def partial_fit(self, train_main, train_appliances, pretrain = False, do_preprocessing=True,pretrain_path = "./dae_pre_state_dict.pkl",**load_kwargs):         
        # If no appliance wise parameters are specified, then they are computed from the data
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        # print(train_main[0].shape) # (1440, 2)
        # Preprocess the data and bring it to a valid shape
        if do_preprocessing:
            print ("Doing Preprocessing")
            train_main,train_appliances = self.call_preprocessing(train_main,train_appliances,'train')
        # print(train_main.shape)
        train_main = np.concatenate(train_main, axis = 0)
        # train_main = train_main.reshape((-1,self.sequence_length,1))

        new_train_appliances  = []
        for app_name, app_arr in train_appliances:
            app_arr = np.concatenate(app_arr, axis=0)
            # print(app_arr.shape)
            # app_df = app_df.reshape((-1,self.sequence_length,1))
            new_train_appliances.append((app_name, app_arr))
        train_appliances = new_train_appliances
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print ("First model training for ",appliance_name)
                self.models[appliance_name] = DAE_Pytorch_MultiChannel(self.sequence_length, train_main.shape[2])
                # Load pretrain dict or not
                if pretrain is True:
                    self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_mul_dae_pre_state_dict.pt"))
  
            model = self.models[appliance_name]
            train(appliance_name, model, train_main, power, self.n_epochs, self.batch_size, pretrain)
            # Model test will be based on the best model
            self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_mul_dae_best_state_dict.pt"))


    def disaggregate_chunk(self, test_main_list, do_preprocessing = True):
        # Disaggregate (test process)
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list,submeters_lst = None,method='test')

        test_predictions = []
        for test_main in test_main_list:
            # test_main = test_main.values.reshape((-1,self.sequence_length,1))
            disggregation_dict = {}

            for appliance in self.models:
                # Move the model to cpu, and then test it
                model = self.models[appliance].to('cpu')
                prediction = test(model, test_main)
                app_mean, app_std = self.appliance_params[appliance]['mean'], self.appliance_params[appliance]['std']
                prediction = self.denormalize_output(prediction,app_mean,app_std)
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                series = pd.Series(valid_predictions)
                disggregation_dict[appliance] = series
            results = pd.DataFrame(disggregation_dict, dtype = 'float32')
            test_predictions.append(results)
        return test_predictions

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        # Seq2Seq Version
        sequence_length  = self.sequence_length
        if method=='train':
            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            processed_mains = []
            for mains in mains_lst:
                self.mains_mean, self.mains_std = mains.values.mean(), mains.values.std()               
                mains = self.normalize_data(mains.values,sequence_length, True)
                processed_mains.append(mains)

            tuples_of_appliances = []
            for (appliance_name,app_df_list) in submeters_lst:
                app_mean = self.appliance_params[appliance_name]['mean']
                app_std = self.appliance_params[appliance_name]['std']
                processed_app_dfs = []
                for app_df in app_df_list:
                    data = self.normalize_data(app_df.values, sequence_length, True)
                    processed_app_dfs.append(data)                    
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances

        if method=='test':
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            processed_mains = []
            for mains in mains_lst:                
                mains = self.normalize_data(mains.values,sequence_length, False)
                processed_mains.append(mains)
            return processed_mains
        
    def normalize_data(self,data,sequence_length, overlapping = False):
        # If you want to train the model,then overlapping = True will bring you a lot more training data; else overlapping = false to disaggregate the mains data
        n = sequence_length
        excess_entries =  sequence_length - (data.shape[0] % sequence_length) 
        windowed_x_all = []
        for i in range(data.shape[1]):
            lst = np.array([0] * excess_entries)
            arr = np.concatenate((data[:,i].flatten(), lst), axis = 0) 
            mean, std = np.mean(data[:,i]), np.std(data[:,i])
            if overlapping:
                windowed_x = np.array([arr[j:j + n] for j in range(len(arr)-n+1)])
            else:
                windowed_x = arr.reshape((-1,sequence_length))
            # print(windowed_x.shape)
            windowed_x = (windowed_x - mean) / std
            windowed_x_all.append(windowed_x.reshape(-1, n, 1))
        windowed_x_all = np.concatenate(windowed_x_all, axis = 2)
        return  windowed_x_all

    def denormalize_output(self,data,mean,std):
        # x = y * std + mean
        return mean + data * std
    
    def set_appliance_params(self,train_appliances):
        # Set appliance mean and std to normalize the label(appliance data)
        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
