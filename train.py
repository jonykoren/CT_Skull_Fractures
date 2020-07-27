# Import Libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve , auc

from PIL import Image
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Specify transforms using torchvision.transforms as transforms
transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



# Load in each dataset and apply transformations using
# the torchvision.datasets as datasets library
train_set = datasets.ImageFolder("/homedtic/ikoren/skull/dat/train", transform = transformations)
val_set = datasets.ImageFolder("/homedtic/ikoren/skull/dat/test", transform = transformations)



# Put into a Dataloader using torch library
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)


# Get pretrained model using torchvision.models as models library
model = models.densenet161(pretrained=True)
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False


# Create new classifier for model using torch.nn as nn library

# Initialize classifier
classifier_input = model.classifier.in_features

# number of classes
num_labels = 2 

classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))

# Replace default classifier with new classifier
model.classifier = classifier



# Find the device available to use using torch library
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device specified above
model.to(device)

# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()

# Set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.classifier.parameters())


# Train Model
def train_net(epochs):
    lossiloss = []
    valiloss = []
    acc = []
    epoc = []
    tmp=[]
    o1 = []
    o2 = []
    for epoch in range(1,epochs+1):
        train_loss = 0
        val_loss = 0
        accuracy = 0

        # Training the model
        model.train()
        counter = 0
        print("\n\nEpoch ", epoch," - Training\n=====================")
        for inputs, labels in train_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)        # Clear optimizers
            optimizer.zero_grad()        # Forward pass
            output = model.forward(inputs)        # Loss
            loss = criterion(output, labels)        # Calculate gradients (backpropogation)
            loss.backward()        # Adjust parameters based on gradients
            optimizer.step()        # Add the loss to the training set's rnning loss
            train_loss += loss.item()*inputs.size(0)

            # Print the progress of our training
            counter += 1
            #print("Epoch ", epoch," - Training")
            sys.stdout.write(f"Step {counter:1d} / {len(train_loader):1d}\r")
            sys.stdout.flush()
            #sys.stdout.write(f"Fitting {clf_amount:1d}/{len(alphas)*n_trials:1d} Multi-Layer-Perceptron classifiers\r")
        # Evaluating the model
        model.eval()
        counter = 0
        print("Epoch ", epoch," - Validation\n=====================")
        # Tell torch not to calculate gradients
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move to device
                inputs, labels = inputs.to(device), labels.to(device)            # Forward pass
                output = model.forward(inputs)            # Calculate Loss
                valloss = criterion(output, labels)            # Add loss to the validation set's running loss
                val_loss += valloss.item()*inputs.size(0)
                
                tmp.append(val_loss)
                
                # Since our model outputs a LogSoftmax, find the real 
                # percentages by reversing the log function
                output = torch.exp(output)            # Get the top class of the output
                top_p, top_class = output.topk(1, dim=1)            # See how many of the classes were correct?
                equals = top_class == labels.view(*top_class.shape)            # Calculate the mean (get the accuracy for this batch)
                # and add it to the running accuracy for this epoch
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Print the progress of our evaluation
                counter += 1
                #print("Epoch ", epoch,"\n===============\nValidation\n===============\n")
                sys.stdout.write(f"Step {counter:1d} / {len(val_loader):1d}\r")
                sys.stdout.flush()  
        #print(tmp)
        # Get the average loss for the entire epoch
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = val_loss/len(val_loader.dataset)    
        ac = accuracy/len(val_loader)
        # Print out the information
        print('Accuracy: ', ac) # accuracy on val set
        print('Training Loss: {:.6f} \nValidation Loss: {:.6f}'.format(train_loss, valid_loss),'\n')

        lossiloss.append(train_loss)
        valiloss.append(valid_loss)
        acc.append(ac)
        epoc.append(epoch)
        
    return lossiloss , valiloss , acc , epoc # Loss, Val Loss, Accuracy, number of epochs



# Loss, Val Loss, Accuracy, number of epochs
lossiloss , valiloss , acc , epoc = train_net(epochs = 10) # change here the number of epochs

model.eval()


# save model for future fast inference
PATH = '/homedtic/ikoren/skull/nuevo/ok/model_outputs/mymodel.pth'
torch.save(model.state_dict(), PATH)
print("model saved successfuly!")

# Loss, Val Loss, Accuracy, number of epochs
np.save("/homedtic/ikoren/skull/nuevo/ok/model_outputs/lossiloss.npy" ,  lossiloss)
np.save("/homedtic/ikoren/skull/nuevo/ok/model_outputs/valiloss.npy" , valiloss)
np.save("/homedtic/ikoren/skull/nuevo/ok/model_outputs/acc.npy" , acc)
np.save("/homedtic/ikoren/skull/nuevo/ok/model_outputs/epoc.npy" , epoc)