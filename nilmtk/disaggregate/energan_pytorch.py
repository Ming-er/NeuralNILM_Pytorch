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
from torchsummary import summary
import torch
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

class Power_Generator(nn.Module):
    # Refer to "KASELIMI M, VOULODIMOS A, PROTOPAPADAKIS E, et al. EnerGAN: A GENERATIVE ADVERSARIAL NETWORK FOR ENERGY DISAGGREGATION[C].2020 ICASSP"
    def __init__(self):
        super(Power_Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size = 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 128, kernel_size = 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 64, kernel_size = 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 128, kernel_size = 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(128, 256, kernel_size = 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(256, 1, kernel_size = 32),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, power_seq):
        return self.model(power_seq)

class Power_Discriminator(nn.Module):
    # Refer to "KASELIMI M, VOULODIMOS A, PROTOPAPADAKIS E, et al. EnerGAN: A GENERATIVE ADVERSARIAL NETWORK FOR ENERGY DISAGGREGATION[C].2020 ICASSP"
    def __init__(self, sequence_length):
        super(Power_Discriminator, self).__init__()
        self.sequence_length = sequence_length
        self.model = nn.Sequential(
            nn.Conv1d(2, 512, kernel_size = 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 256, kernel_size = 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear((self.sequence_length - 46)*256, 60),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(60, 1),
            nn.Sigmoid()
        )

    def forward(self, mains_seq, app_seq):
        mains_app = torch.cat((mains_seq, app_seq), dim = 1)
        validity = self.model(mains_app)
        return validity

def initialize(layer):
    # Xavier_uniform will be applied to conv1d, convtranspose1d and dense layer, to be consistent with Keras and Tensorflow
    if isinstance(layer,nn.Conv1d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.ConvTranspose1d):    
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, val = 0.0)

def train(appliance_name, gen_model, dis_model, mains, appliance, epochs, batch_size, pretrain = False, checkpoint_interval = None, train_patience = 3):
    # Model configuration
    if USE_CUDA:
        gen_model = gen_model.cuda()
        dis_model = dis_model.cuda()
    if not pretrain:
        gen_model.apply(initialize)
        dis_model.apply(initialize)
    summary(gen_model, (1, mains.shape[1]))
    summary(dis_model, [(1, mains.shape[1]), (1, mains.shape[1])])

    # Split the train and validation set
    train_mains,valid_mains,train_appliance,valid_appliance = train_test_split(mains, appliance, test_size=.2, random_state = random_seed)

    # Create optimizer, loss function, and dataload
    optimizer_G = torch.optim.Adam(gen_model.parameters(), lr = 1e-3)
    optimizer_D = torch.optim.Adam(dis_model.parameters(), lr = 1e-3)

    adversarial_loss = torch.nn.BCELoss()
    reconstruction_loss = torch.nn.L1Loss()
    eval_loss = torch.nn.MSELoss()

    train_dataset = TensorDataset(torch.from_numpy(train_mains).float().permute(0,2,1), torch.from_numpy(train_appliance).float().permute(0,2,1))
    train_loader = tud.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)

    valid_dataset = TensorDataset(torch.from_numpy(valid_mains).float().permute(0,2,1), torch.from_numpy(valid_appliance).float().permute(0,2,1))
    valid_loader = tud.DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)

    writer = SummaryWriter(comment='train_visual')
    patience, best_loss = 0, None

    for epoch in range(epochs):
        # Earlystopping
        if(patience == train_patience):
            print("val_loss did not improve after {} Epochs, thus Earlystopping is calling".format(train_patience))
            break   
        # Train the model(adversarial)
        st = time.time() 
            
        for i, (batch_mains, batch_appliance) in enumerate(train_loader):

            valid = torch.ones(batch_size, 1).float()
            fake = torch.zeros(batch_size, 1).float()

            if USE_CUDA:
                batch_mains = batch_mains.cuda()
                batch_appliance = batch_appliance.cuda()
                valid = valid.cuda()
                fake = fake.cuda()

            # Generator Training
            gen_model.train()

            batch_pred = gen_model(batch_mains)
            fake_labels = dis_model(batch_pred, batch_mains)
            '''
            Please notice that our implementation is slightly different from the original paper, since we incorporate reconstruction loss(L1 loss)
            while training Generator, which is proved to be valid in most pix2pix task.And our approach will result in improved accuracy.
            '''
            ad_loss = adversarial_loss(fake_labels, valid)
            pix_loss = reconstruction_loss(batch_pred, batch_appliance)
            gen_loss = ad_loss + 100 * pix_loss

            optimizer_G.zero_grad()    
            gen_loss.backward()
            optimizer_G.step()

            # Discriminator Training
            dis_model.train()

            true_labels = dis_model(batch_appliance, batch_mains)
            fake_labels = dis_model(batch_pred.detach(), batch_mains)
            
            real_loss = adversarial_loss(true_labels, valid)
            fake_loss = adversarial_loss(fake_labels, fake)
            dis_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()    
            dis_loss.backward()
            optimizer_D.step()


        ed = time.time()

        # Evaluate the model
        gen_model.eval()

        with torch.no_grad():
            cnt, loss_sum = 0, 0
            for i, (batch_mains, batch_appliance) in enumerate(valid_loader):
                if USE_CUDA:
                    batch_mains = batch_mains.cuda()
                    batch_appliance = batch_appliance.cuda()
            
                batch_pred = gen_model(batch_mains)
                loss = eval_loss(batch_appliance, batch_pred)
                loss_sum += loss
                cnt += 1        

        final_loss = loss_sum / cnt
        # Save best only 
        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            patience = 0
            gennet_state_dict = gen_model.state_dict()
            path_state_dict = "./"+appliance_name+"_energan_best_state_dict.pt"
            torch.save(gennet_state_dict, path_state_dict)
        else:
            patience = patience + 1 
        print("Epoch: {}, Valid_Loss: {}, Time consumption: {}s.".format(epoch, final_loss, ed - st))
        # For the visualization of training process
        for name, param in gen_model.named_parameters():
            writer.add_histogram(name + '_grad', param.grad, epoch)
            writer.add_histogram(name + '_data', param, epoch)
        writer.add_scalars("MSELoss", {"Valid":final_loss}, epoch)
        # Save checkpoint
        if (checkpoint_interval != None) and ((epoch + 1) % checkpoint_interval == 0):
            checkpoint = {"genmodel_state_dict": gen_model.state_dict(),
                            "dismodel_state_dict": dis_model.state_dict(),
                            "optimizerG_state_dict": optimizer_G.state_dict(),
                            "optimizerD_state_dict": optimizer_D.state_dict(),
                            "epoch": epoch}
            path_checkpoint = "./"+appliance_name+"_energan_checkpoint_{}_epoch.pt".format(epoch)
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

class EnerGAN(Disaggregator):
    
    def __init__(self, params):
        self.MODEL_NAME = "EnerGAN"
        self.sequence_length = params.get('sequence_length',63)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size',32)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',None)
        self.mains_std = params.get('mains_std',None)
        self.models = OrderedDict()       

        
    def partial_fit(self, train_main, train_appliances,pretrain = False, do_preprocessing=True,**load_kwargs):        
        # Seq2Seq version
        # If no appliance wise parameters are specified, then they are computed from the data
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        # To preprocess the data and bring it to a valid shape
        if do_preprocessing:
            print ("Doing Preprocessing")
            train_main,train_appliances = self.call_preprocessing(train_main,train_appliances,'train')

        train_main = pd.concat(train_main,axis = 0).values
        train_main = train_main.reshape((-1,self.sequence_length,1))

        # train_main.shape = (total_batch,seqlen,1)
        new_train_appliances  = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df,axis=0).values
            app_df = app_df.reshape((-1,self.sequence_length,1))
            new_train_appliances.append((app_name, app_df))
        train_appliances = new_train_appliances
        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print ("First model training for ",appliance_name)
                self.models[appliance_name] = Power_Generator()  
                gen_model = self.models[appliance_name]
                dis_model = Power_Discriminator(self.sequence_length)
                # Load pretrain dict or not
                if pretrain:
                    self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_genmodel_pre_state_dict.pt"))
                    dis_model.load_state_dict(torch.load("./"+appliance_name+"_dismodel_pre_state_dict.pt"))

            train(appliance_name,gen_model, dis_model, train_main, power, self.n_epochs, self.batch_size, pretrain, checkpoint_interval = 3)
            # Model test will be based on the best model
            self.models[appliance_name].load_state_dict(torch.load("./"+appliance_name+"_energan_best_state_dict.pt"))

    def disaggregate_chunk(self, test_main_list, do_preprocessing = True):
        # Disaggregate (test process)
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list,submeters_lst = None,method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values.reshape((-1,self.sequence_length,1))
            disggregation_dict = {}

            for appliance in self.models:
                # Move the model to cpu, and then test it
                model = self.models[appliance].to('cpu')
                prediction = test(model, test_main)
                app_mean, app_std = self.appliance_params[appliance]['mean'], self.appliance_params[appliance]['std']
                prediction = self.denormalize_output(prediction,app_mean,app_std)
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0,valid_predictions,0)
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
                self.mains_mean, self.mains_std = mains.values.mean(), mains.values.std()               
                mains = self.normalize_data(mains.values,sequence_length, mains.values.mean(), mains.values.std(),True)
                processed_mains.append(pd.DataFrame(mains))

            tuples_of_appliances = []
            for (appliance_name,app_df_list) in submeters_lst:
                app_mean = self.appliance_params[appliance_name]['mean']
                app_std = self.appliance_params[appliance_name]['std']
                processed_app_dfs = []
                for app_df in app_df_list:
                    data = self.normalize_data(app_df.values, sequence_length,app_mean,app_std,True)
                    processed_app_dfs.append(pd.DataFrame(data))                    
                tuples_of_appliances.append((appliance_name, processed_app_dfs))

            return processed_mains, tuples_of_appliances

        if method=='test':
            # Preprocess the main data only, the parameter 'overlapping' will be set 'False'
            processed_mains = []
            for mains in mains_lst:                
                mains = self.normalize_data(mains.values,sequence_length, mains.values.mean(), mains.values.std(),False)
                processed_mains.append(pd.DataFrame(mains))
            return processed_mains
        
    def normalize_data(self,data,sequence_length, mean, std, overlapping = False):
        # If you want to train the model,then overlapping = True will bring you a lot more training data; else overlapping = false to disaggregate the mains data
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0)   
        if overlapping:
            windowed_x = np.array([arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_x = arr.reshape((-1,sequence_length))
        windowed_x = windowed_x - mean
        # z-score normalization: y = (x - mean)/std
        return (windowed_x / std).reshape((-1,sequence_length))

    def denormalize_output(self,data,mean,std):
        # x = y * std + mean
        return mean + data * std
    
    def set_appliance_params(self,train_appliances):
        # Set appliance mean and std to normalize the label(appliance data)
        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
