# Python3 Code
# Three Water's Coding Style
# edited and tested by Akida-Sho Wong
#
# log
#
# @Oct 3 11:19:18am begins
import os
from tqdm import tqdm

import random
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

# modules
from model import FedPerModel
from dataSplit import get_data_loaders
from utils import AverageMeter
# from sampling import uniform_sampling
from LossRatio import get_auxiliary_data_loader, compute_ratio_per_client_update
from scipy.stats import entropy

#CHANGE: Added preciison, recall and f1 score import
from sklearn.metrics import precision_score, recall_score, f1_score

# GPU settings
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

NUM_CLASSES = 10
# Parameters that can be tuned during different simulations
num_clients = 100
num_selected = 20
num_rounds = 2020
#num_rounds = 3150	# num of communication rounds

epochs = 5			# num of epochs in local client training (An epoch means you go through the entire dataset in that client)
batch_size = 10 # batch size already set in train dataloaderin dataSplit.py
num_batch = 10 # change here choose 100 - 300 default:50

# hyperparameters of deep models
lr = 0.1 # learning rate
decay_factor = 0.996

losses_train = []
losses_test = []
acc_train = []
acc_test = []
client_idx_lst = []
T_pull_lst = []



alpha = 0.8
# cucb parameters
reward_est = np.zeros(num_clients)
reward_bias = np.zeros(num_clients)
T_pull = np.zeros(num_clients)
reward_global = np.zeros(num_rounds)
reward_client = np.zeros(num_clients)

# V_pt statistic
V_pt_avg = np.zeros((num_clients, NUM_CLASSES))

def traverse(N,K,RR):
    '''sampling each client at least ones'''
    R = int(np.ceil(N/K))
    R_set = np.arange(R)

    selected_set = np.zeros(K)

    if RR in R_set:
        idx = np.where(R_set == RR)[0][0]
        selected_set = np.arange(idx*K, (idx+1)*K)
    for i in range(K):
        if selected_set[i] >= N:
            selected_set[i] = selected_set[i] - N
    return selected_set

def cross_entropy(y):
    x = np.ones(y.shape)
    return entropy(x) + entropy(x, y + 1e-15)

def get_client_set(reward, K, N, V_pt_avg):
    '''get client set for cucb'''
    # create dict
    V_pt_dict = {}
    for i in range(N):
        V_pt_dict[i] = V_pt_avg[i,:]
    # choose the max reward client as base
    r_max_index = np.argmax(reward)
    # remove the min index
    V_pt_dict.pop(r_max_index)
    # combination index set S
    S = np.array(r_max_index)
    # combination distribution set
    comb_set = np.array(V_pt_avg[r_max_index])

    while S.size < K:
        ce_reward_set = {}
        for key,value in V_pt_dict.items():
            # calculate the avg class distribution
            comb_dist = np.vstack([comb_set, V_pt_dict[key]])
            comb_dist_avg = np.sum(comb_dist, axis=0) / comb_dist.shape[0]
            # calculate cos loss of combined distribution
            ce_loss = cross_entropy(comb_dist_avg)
            ce_reward_set[key] = 1 / ce_loss

        # get the cos ratio loss index
        reward_max_idx = max(V_pt_dict.keys(),key=(lambda x:ce_reward_set[x]))

        # remove the selected client

        S = np.append(S, reward_max_idx)
        comb_set = np.vstack([comb_set, V_pt_dict[reward_max_idx]])
        V_pt_dict.pop(reward_max_idx)

    return S

# MODIFIED: To use FedPer and add personalisation layers
def main():
    # Data loader
    client_train_loader, test_loader, data_size_per_client = get_data_loaders(num_clients, batch_size, True)
    data_size_weights = data_size_per_client / data_size_per_client.sum()
    
    aux_loader = get_auxiliary_data_loader(testset_extract=True, data_size=32)
    
    print("Test dataset length:", len(test_loader.dataset))
    for batch_idx, (data, target) in enumerate(test_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        print("Data min/max:", data.min().item(), data.max().item())
        print("Targets sample:", target[:10].tolist())
        if batch_idx >= 2:  # Check first 3 batches only
            break

    # Model configurations: global and client models
    global_model = FedPerModel().to(device)
    client_models = [FedPerModel().to(device) for _ in range(num_selected)]
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    # Client optimizers
    opt_lst = [optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4) for model in client_models]
    
    # For each communication round
    for r in range(num_rounds):
        #print(lr * (decay_factor ** r))
        
        for opt in opt_lst:
            opt.param_groups[0]['lr'] = lr * (decay_factor ** r)

        #MODIFIED: To include random selection for baseline model comparison
        use_random_selection = False  

        if use_random_selection:
            client_idx = random.sample(range(num_clients), num_selected)
        else:
            RR = int(np.ceil(num_clients / num_selected))
            if r < RR:
                client_idx = traverse(num_clients, num_selected, r)
            else:
                reward_bias = reward_client + alpha * np.sqrt(3 * np.log(r) / (2 * T_pull))
                client_idx = get_client_set(reward_bias, num_selected, num_clients, V_pt_avg)
        for s in client_idx:
            T_pull[s] += 1


        loss = 0
        for i in range(num_selected):
            loss += client_update(client_models[i], opt_lst[i], client_train_loader[client_idx[i]], epochs)
            
        ra_dict = compute_ratio_per_client_update(client_models, client_idx, aux_loader)
        for i in range(num_selected):
            reward_single, V_pt = 1 / cross_entropy(ra_dict[client_idx[i]]), ra_dict[client_idx[i]]
            reward_client[client_idx[i]] = (reward_client[client_idx[i]] * (T_pull[client_idx[i]] - 1) + reward_single) / T_pull[client_idx[i]]
            V_pt_avg[client_idx[i]] = (V_pt_avg[client_idx[i]] * (T_pull[client_idx[i]] - 1) + V_pt) / T_pull[client_idx[i]]
        
        reward_global[r] = 1 / cross_entropy(compute_ratio_per_client_update([global_model], client_idx, aux_loader)[client_idx[0]])
        losses_train.append(loss / num_selected)

        # Step 4: Updated local models send back for server aggregate
        server_aggregate(global_model, client_models, data_size_weights, client_idx)

        # Global Evaluation and Personalised Evaluation are performed only every 20 rounds
        if r % 20 == 0:
            print('%d-th round' % r)
            # Global evaluation with extended metrics
            global_test_loss, global_acc, global_prec, global_rec, global_f1 = evaluate_model_with_metrics(global_model, test_loader)
            print('Global evaluation - Loss: {:.3g}, Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(
                global_test_loss, global_acc, global_prec, global_rec, global_f1))
            
            # Per-class evaluation:
            per_class_precision, per_class_recall, per_class_f1 = evaluate_per_class_metrics(global_model, test_loader, device)
            for cls in range(len(per_class_precision)):
                print(f"Global - Class {cls} - Precision: {per_class_precision[cls]:.3f}, Recall: {per_class_recall[cls]:.3f}, F1: {per_class_f1[cls]:.3f}")
            
            # Personalised Evaluation with additional metrics
            personalized_metrics = []
            for i in range(num_selected):
                fine_tune_personalized_layer(client_models[i], client_train_loader[client_idx[i]], num_epochs=1, lr=0.001)
                ploss, pacc, pprec, precall, pf1 = evaluate_model_with_metrics(client_models[i], test_loader)
                personalized_metrics.append((ploss, pacc, pprec, precall, pf1))
            avg_ploss = np.mean([m[0] for m in personalized_metrics])
            avg_pacc = np.mean([m[1] for m in personalized_metrics])
            avg_pprec = np.mean([m[2] for m in personalized_metrics])
            avg_precal = np.mean([m[3] for m in personalized_metrics])
            avg_pf1 = np.mean([m[4] for m in personalized_metrics])
            print('Personalized evaluation - Avg Loss: {:.3g}, Avg Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(
                avg_ploss, avg_pacc, avg_pprec, avg_precal, avg_pf1))
            print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss / num_selected, global_test_loss, global_acc))


        T_pull_lst.append(np.copy(T_pull))
        client_idx_lst.append(client_idx)
        name = "log_lr"+str(lr)+"_decay"+str(decay_factor)+"_alpha"+str(alpha)+"_C"+str(num_clients)+"_S"+str(num_selected)+"_Nbatch"+str(num_batch)+".mat"
        sio.savemat(name, {'V_pt_avg': V_pt_avg, 'reward_client': reward_client, 'reward_global': reward_global, 'acc_test': acc_test, 'T_pull': T_pull_lst, 'client_idx': client_idx_lst})

# actually standard training in a local client/device
def client_update(client_model, optimizer, train_loader, epochs):
    loss_avg = AverageMeter()

    client_model.train()

    for e in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == num_batch:
                break

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = client_model(data)
            loss = F.cross_entropy(output, target)
            loss_avg.update(loss.item(), data.size(0))

            loss.backward()

            # Only update personalized layers
            for param in client_model.personalized_model.parameters():
                param.grad = None  # Zero out gradients for personalized layers

            optimizer.step()

    return loss_avg.avg # average loss in this client over entire trainset over multiple epochs

#MODIFIED: Updates only the shared layers
def server_aggregate(global_model, client_models, data_size_weights, client_idx):
    # Get the state_dict of the global model
    global_state = global_model.state_dict()
    
    # Re-normalize the weights for the selected clients
    selected_weights = data_size_weights[client_idx]
    weight_sum = selected_weights.sum()
    
    # Aggregate only parameters for the global model.
    for k in global_state.keys():
        if k.startswith("global_model."):
            # Compute the weighted average of the parameter
            aggregated = torch.stack([
                selected_weights[i] * client_models[i].state_dict()[k].float()
                for i in range(len(client_models))
            ], 0).sum(0) / weight_sum  # Divide by the sum of the selected weights
            global_state[k] = aggregated
        # For personalized parameters, we leave the global model's version unchanged.
    
    # Load the aggregated global parameters into the global model.
    global_model.load_state_dict(global_state)
    
    # Update each client model with the new global parameters, preserving their personalized layers.
    for model in client_models:
        client_state = model.state_dict()
        for k in client_state.keys():
            if k.startswith("global_model."):
                client_state[k] = global_state[k]
        model.load_state_dict(client_state)



def test(model, test_loader):
    total_correct = 0
    total_samples = 0
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Ensure you're using the full model if needed
            loss = F.cross_entropy(output, target)
            loss_sum += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
    avg_loss = loss_sum / total_samples
    accuracy = (total_correct / total_samples) * 100  # Percentage
    return avg_loss, accuracy

def fine_tune_personalized_layer(model, fine_tune_loader, num_epochs=1, lr=0.001):
    """
    Fine-tune only the personalised layer on the provided fine-tuning data.
    This function freezes the global layers and trains the personalised layer.
    """
    for param in model.global_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.personalized_model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        for data, target in fine_tune_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    for param in model.global_model.parameters():
        param.requires_grad = True

    return model

def evaluate_personalized(model, test_loader):
    """
    Evaluate the full model (global + personalized layers) after fine-tuning.
    """
    model.eval()
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss_avg.update(loss.item(), data.size(0))
            pred = output.argmax(dim=1, keepdim=True)
            acc_avg.update(pred.eq(target.view_as(pred)).sum().item(), data.size(0))
    return loss_avg.avg, acc_avg.avg


#CHANGE: Added evaluation for precision,recall and f1 scores
def evaluate_model_with_metrics(model, test_loader):
    model.eval()
    loss_sum = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss_sum += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            total_correct += preds.eq(target).sum().item()
            total_samples += data.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = loss_sum / total_samples
    accuracy = (total_correct / total_samples) * 100
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, precision, recall, f1

#MODIFIED: Including metrics for each class
def evaluate_per_class_metrics(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    per_class_precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    
    return per_class_precision, per_class_recall, per_class_f1

if __name__ == '__main__':
	main()
