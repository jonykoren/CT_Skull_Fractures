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

# Load model from path
PATH = '/homedtic/ikoren/skull/nuevo/ok/model_outputs/mymodel.pth'

# Load model - starting inference
net = model
net.load_state_dict(torch.load(PATH))
dataiter = iter(val_loader)
images, labels = dataiter.next()
images = images.cuda()
labels = labels.cuda()
outputs = net(images)
classes = ('b', 'nb')

# Load probs. 
lossiloss = np.load("/homedtic/ikoren/skull/nuevo/ok/model_outputs/lossiloss.npy")
valiloss = np.load("/homedtic/ikoren/skull/nuevo/ok/model_outputs/valiloss.npy")
acc = np.load("/homedtic/ikoren/skull/nuevo/ok/model_outputs/acc.npy")
epoc = np.load("/homedtic/ikoren/skull/nuevo/ok/model_outputs/epoc.npy")

_, predicted = torch.max(outputs, 1)



plt.figure(figsize=(20,10))


def imshow(img):
    img = img / 2 + 0.5  # unnormalize image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("/homedtic/ikoren/skull/nuevo/ok/plots/pretrained_imgs.png")
    #plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# Print total validation accuracy of the model
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


# Classes accuracy
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(classes)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# Print info classes
#for i in range(len(classes)):
#    print('Accuracy of %5s : %2d %%' % (
#        classes[i], 100 * class_correct[i] / class_total[i]))

# Loss plot
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.plot(epoc,lossiloss, color="red")
plt.plot(epoc,valiloss, color="blue")
plt.legend(['Loss', 'Val Loss'], loc='upper right')
plt.title("Train Loss", size=28)
plt.xlabel("Epochs", size=20)
plt.ylabel("Loss", size=20)
plt.savefig("/homedtic/ikoren/skull/nuevo/ok/plots/loss.png")
#plt.show()

# Accuracy plot
plt.style.use('ggplot')
plt.figure(figsize=(20,10))
plt.plot(epoc,acc, color="blue")
plt.legend(['Train Accuracy'], loc='upper left')
plt.title("Accuracy", size=28)
plt.xlabel("Epochs", size=20)
plt.ylabel("Accuracy", size=20)
plt.savefig("/homedtic/ikoren/skull/nuevo/ok/plots/accuracy.png")
#plt.show()


# Confusion Matrix
def test_label_predictions(model, device, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]

actuals, predictions = test_label_predictions(model, device, val_loader)
print('Confusion matrix:\n==================')
print(confusion_matrix(actuals, predictions))
print()
print('F1 score: %f' % f1_score(actuals, predictions, average='micro'))
print("=========")
print()
print('Accuracy score: %f' % accuracy_score(actuals, predictions))
print("===============")


# Class Definition
dataset = datasets.ImageFolder('/homedtic/ikoren/skull/dat/train', transform=transformations)
classes1 = dataset.class_to_idx

# Get probabilities - test , predictions
def test_class_probabilities(model, device, test_loader, which_class):
    model.eval()
    model.cpu()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.cpu()
            target = target.cpu()
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction) == which_class)
            probabilities.extend(np.exp(output[:, which_class]))
    return [i.item() for i in actuals], [i.item() for i in probabilities]



which_class = 1 # non-broken 

actuals, class_probabilities = test_class_probabilities(model, device, val_loader, which_class)

# save probs. for future fast inference
np.save("/homedtic/ikoren/skull/nuevo/ok/model_outputs/class_probabilities_nb.npy", class_probabilities)
np.save("/homedtic/ikoren/skull/nuevo/ok/model_outputs/actuals_nb.npy", actuals)

fpr, tpr, _ = roc_curve(actuals, class_probabilities)
roc_auc = auc(fpr, tpr)

# plt config
plt.style.use('ggplot')
plt.figure(figsize=(20,10))

lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size=20)
plt.ylabel('True Positive Rate', size=20)
plt.title('ROC for non-broken class', size=28)
plt.legend(loc="lower right")
plt.savefig("/homedtic/ikoren/skull/nuevo/ok/plots/roc_nb.png")
#plt.show()


which_class = 0 # broken 
actuals, class_probabilities = test_class_probabilities(model, device, val_loader, which_class)

# save probs. for future fast inference
np.save("/homedtic/ikoren/skull/nuevo/ok/model_outputs/class_probabilities_b.npy", class_probabilities)
np.save("/homedtic/ikoren/skull/nuevo/ok/model_outputs/actuals_b.npy", actuals)

fpr, tpr, _ = roc_curve(actuals, class_probabilities)
roc_auc = auc(fpr, tpr)

# plt config
plt.style.use('ggplot')
plt.figure(figsize=(20,10))

lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate' , size = 20)
plt.ylabel('True Positive Rate' , size = 20)
plt.title('ROC for broken class' , size = 28)
plt.legend(loc="lower right")
plt.savefig("/homedtic/ikoren/skull/nuevo/ok/plots/roc_b.png")
#plt.show()

