import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MyDataLoader import MyNoiseDataset, MyNoiseDataset1
from M5_Network import m3, m5, m11, m18, m34_res, m6_res

BATCH_SIZE = 250

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size)
    return train_dataloader

#-------------------------------------------------------------
# Function  :   load_weigth_for_model()
# Loading pre-trained weights to model
#-------------------------------------------------------------
def load_weigth_for_model(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location="cuda:0")
    for k, v in model_dict.items():
        model_dict[k] = pretrained_dict[k]
    model.load_state_dict(model_dict)

#-------------------------------------------------------------
# Function : validate_single_epoch()
# Testing the accuracy of the trained model in test dataset.
#-------------------------------------------------------------
def validate_single_epoch(model, eva_data_loader, device):
    eval_loss = 0 
    eval_acc = 0 
    model.eval()
    i = 0
    
    for input, target in eva_data_loader:
        input, target = input.to(device), target.to(device)
        i += 1 
        
        # Calculating the loss value
        prediction = model(input)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(prediction,target)

        # recording the validating loss and accuratcy
        eval_loss += loss.item()
        _, pred = prediction.max(1)
        num_correct = (pred == target).sum().item()
        acc = num_correct / input.shape[0]
        eval_acc += acc

    print(f"Test loss: {eval_loss/i}" + f"Test accuracy: {eval_acc/i}") 
    return eval_acc/i

#----------------------------------------------------------------------------------------
# Function : Testing the accuracy of the trained model in the testing set.
#----------------------------------------------------------------------------------------
def Test_model_accuracy_original(TESTING_DATASET_FILE, MODLE_PTH, File_sheet):
    testing_dataset = MyNoiseDataset(TESTING_DATASET_FILE, File_sheet)
    testing_loader = create_data_loader(testing_dataset, int(BATCH_SIZE/10))
    
    # set the model
    model = m6_res
    device = torch.device('cuda')
    model = model.to(device)
    
    # loading coefficients 
    load_weigth_for_model(model, MODLE_PTH)
    
    # testing model
    accuracy = validate_single_epoch(model, testing_loader, device)
    
    return accuracy

def Output_Test_Error_Samples(TESTING_DATASET_FILE, MODLE_PTH, File_sheet):
    testing_dataset = MyNoiseDataset1(TESTING_DATASET_FILE, File_sheet)
    
    # set the model
    model = m6_res
    device = torch.device('cuda')
    model = model.to(device)
    # loading coefficients 
    load_weigth_for_model(model, MODLE_PTH)
    model.eval()
    
    j=0
    #print('length of testing_dataset:%d'%len(testing_dataset))
    for i in range(len(testing_dataset)):
        audio_sample_path, signal, label = testing_dataset[i]
        signal = signal.to(device) # torch.Size([1, 16000])
        signal = signal.unsqueeze(0) # torch.Size([1, 1, 16000])
        prediction = model(signal) # torch.Size([15])
        _, pred = prediction.max(0)
        if pred.item() == label:
            j += 1
        else:
            print(audio_sample_path, pred.item(), label)
    accuracy = j/len(testing_dataset) # test_accuracy
    return accuracy