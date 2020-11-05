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
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
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

class sgn_branch_network(nn.Module):
    def __init__(self, mains_length, appliance_length):
        super(sgn_branch_network, self).__init__()
        self.mains_length = mains_length
        self.appliance_length = appliance_length

        self.conv = nn.Sequential(
            nn.ConstantPad1d((4, 5), 0),
            nn.Conv1d(1, 30, 10, stride = 1),
            nn.ReLU(True),
            nn.ConstantPad1d((3, 4), 0),
            nn.Conv1d(30, 30, 8, stride = 1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 3), 0),
            nn.Conv1d(30, 40, 6, stride = 1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(40, 50, 5, stride = 1),
            nn.ReLU(True),
            nn.ConstantPad1d((2, 2), 0),
            nn.Conv1d(50, 50, 5, stride = 1),
            nn.ReLU(True)
        )

        self.dense = nn.Sequential(
            nn.Linear(50 * self.mains_length, 1024), 
            nn.ReLU(True),
            nn.Linear(1024, self.appliance_length)
        )


    def forward(self, x):
        x = self.conv(x)
        x = self.dense(x.view(-1, 50 * self.mains_length))
        return x.view(-1, self.appliance_length)

class sgn_Pytorch(nn.Module):
    def __init__(self, mains_length, appliance_length):
        # Refer to "SHIN C, JOO S, YIM J. Subtask Gated Networks for Non-Intrusive Load Monitoring[J]. Proceedings of the AAAI Conference on Artificial Intelligence."
        super(sgn_Pytorch, self).__init__()
        self.gate = sgn_branch_network(mains_length, appliance_length)
        self.reg = sgn_branch_network(mains_length, appliance_length)
        self.act = nn.Sigmoid()
        self.b = nn.parameter.Parameter(torch.zeros(1))

    def forward(self, x):
        reg_power = self.reg(x)
        app_state = self.act(self.gate(x))
        app_power = reg_power * app_state + (1 - app_state) * self.b
        return app_power, app_state

def initialize(layer):
    # Xavier_uniform will be applied to conv1d and dense layer, to be consistent with Keras and Tensorflow
    if isinstance(layer,nn.Conv1d) or isinstance(layer, nn.Linear):    
        torch.nn.init.kaiming_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, val = 0.0)

def train(appliance_name,model, mains, appliance, epochs, batch_size, threshold, pretrain, checkpoint_interval = None, train_patience = 3):
    # Model configuration
    if USE_CUDA:
        model = model.cuda()
    if not pretrain:
        model.apply(initialize)
    summary(model,(1, mains.shape[1]))
    # Split the train and validation set
    train_mains,valid_mains,train_appliance,valid_appliance = train_test_split(mains, appliance, test_size=.2, random_state = random_seed)

    # Create optimizer, loss function, and dataloadr
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn_reg = torch.nn.MSELoss()
    loss_fn_cla = torch.nn.BCELoss()

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
        for i, (true_mains_power, true_app_power) in enumerate(train_loader):
            if USE_CUDA:
                true_mains_power = true_mains_power.cuda()
                true_app_power = true_app_power.cuda()
            
            cal_app_state = (true_app_power > threshold).float().cpu().numpy()
            true_app_state = torch.from_numpy(cal_app_state).cuda().detach()
            pred_app_power, pred_app_state = model(true_mains_power)

            loss_reg = loss_fn_reg(pred_app_power, true_app_power)
            loss_cla = loss_fn_cla(pred_app_state, true_app_state)
            loss = loss_reg + loss_cla

            model.zero_grad()    
            loss.backward()
            optimizer.step()

        ed = time.time()
        print("Epoch: {},Time consumption: {}s.".format(epoch, ed - st))

        # Evaluate the model    
        model.eval()
        with torch.no_grad():
            cnt, loss_sum, loss_reg_sum, loss_cla_sum = 0, 0, 0, 0
            for i, (true_mains_power, true_app_power) in enumerate(train_loader):
                if USE_CUDA:
                    true_mains_power = true_mains_power.cuda()
                    true_app_power = true_app_power.cuda()
            
                cal_app_state = (true_app_power > threshold).float().cpu().numpy()
                true_app_state = torch.from_numpy(cal_app_state).cuda()
                pred_app_power, pred_app_state = model(true_mains_power)

                loss_reg = loss_fn_reg(pred_app_power, true_app_power)
                loss_cla = loss_fn_cla(pred_app_state, true_app_state)
                loss = loss_reg + loss_cla

                loss_reg_sum += loss_reg
                loss_cla_sum += loss_cla
                loss_sum += loss

                cnt += 1

        # Save best only
        if best_loss is None or loss_reg_sum / cnt < best_loss:
            best_loss = loss_reg_sum / cnt
            patience = 0
            net_state_dict = model.state_dict()
            path_state_dict = "./"+appliance_name+"_sgn_best_state_dict.pt"
            torch.save(net_state_dict, path_state_dict)
        else:
            patience = patience + 1 

        print("Epoch: {}, Valid_Reg_Loss: {}, Valid_Cla_Loss: {}, Valid_Total_Loss: {}.".format(epoch, loss_reg_sum / cnt, loss_cla_sum / cnt, loss_sum / cnt))
        # For the visualization of training process
        for name,param in model.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, epoch)
            writer.add_histogram(name + '_data', param, epoch)
        writer.add_scalars("MSELoss", {"Valid":loss_reg_sum/cnt}, epoch)

        # Save checkpoint
        if (checkpoint_interval != None) and ((epoch + 1) % checkpoint_interval == 0):
            checkpoint = {"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch}
            path_checkpoint = "./"+appliance_name+"_sgn_checkpoint_{}_epoch.pt".format(epoch)
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
            batch_pred = model(batch_mains[0])[0]
            if i == 0:
                res = batch_pred
            else:
                res = torch.cat((res, batch_pred), dim = 0)
    ed = time.time()
    print("Inference Time consumption: {}s.".format(ed - st))
    return res.numpy()

class SGN(Disaggregator):

    def __init__(self, params):
        self.MODEL_NAME = "SGN"
        self.models = OrderedDict()
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.mains_length = params.get('sequence_length',200)
        self.appliance_length = params.get('appliance_length',32)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',None)
        self.mains_std = params.get('mains_std',None)

    def partial_fit(self,train_main,train_appliances,pretrain = False, do_preprocessing=True,
            **load_kwargs):
        # Seq2Subseq version
        # If no appliance wise parameters are provided, then copmute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        print("...............SGN partial_fit running...............")

        # To preprocess the data and bring it to a valid shape
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1,self.mains_length,1))
        
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0)
            app_df_values = app_df.values.reshape((-1,self.appliance_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = sgn_Pytorch(self.mains_length, self.appliance_length)
                # Load pretrain dict or not
                if pretrain is True:
                    self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_sgn_pre_state_dict.pt"))


            model = self.models[appliance_name]
            train(appliance_name,model, train_main, power, self.n_epochs, self.batch_size, (10.0 - self.appliance_params[app_name]['mean'])/self.appliance_params[app_name]['std'],pretrain, checkpoint_interval = 3)
            # Model test will be based on the best model
            self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_sgn_best_state_dict.pt"))


    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):
        # Disaggregate (test process)
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst = None, method='test')

        test_predictions = []
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            test_main = test_mains_df.values.reshape((-1, self.mains_length, 1))
            for appliance in self.models:
                # Move the model to cpu, and then test it
                model = self.models[appliance].to('cpu')
                predict = test(model, test_main)

                l1 = self.mains_length
                l2 = self.appliance_length
                n = len(predict) + l2 - 1

                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))

                for i in range(predict.shape[0]):
                    sum_arr[i:i + l2] += predict[i].flatten()
                    counts_arr[i:i + l2] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]

                prediction = self.appliance_params[appliance]['mean'] + (sum_arr * self.appliance_params[appliance]['std'])
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)

        return test_predictions

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        # Seq2Subseq Version
        if method == 'train':
            # Preprocess the main and appliance data, the parameter 'overlapping' will be set 'True'
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                self.mains_mean, self.mains_std = new_mains.mean(), new_mains.std()
                n = self.mains_length - self.appliance_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + self.mains_length] for i in range(len(new_mains) - self.mains_length + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
                app_mean, app_std = self.appliance_params[app_name]['mean'], self.appliance_params[app_name]['std']
                
                processed_appliance_dfs = []

                for app_df in app_df_list:
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.array([new_app_readings[i:i + self.appliance_length] for i in range(len(new_app_readings) - self.appliance_length + 1)])
                    new_app_readings = (new_app_readings - app_mean) / app_std  
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_appliance_dfs))
            return mains_df_list, appliance_list

        else:
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            mains_df_list = []

            for mains in mains_lst:
                new_mains = mains.values.flatten()
                self.mains_mean, self.mains_std = new_mains.mean(), new_mains.std()
                n = self.mains_length - self.appliance_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + self.mains_length] for i in range(len(new_mains) - self.mains_length + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def set_appliance_params(self,train_appliances):
        # Set appliance mean and std to normalize the label(appliance data)
        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})