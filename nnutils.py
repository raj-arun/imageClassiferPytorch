import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import time
import torch
import torchvision
import torch.nn.functional as F

from torch import nn, optim
from torch.autograd import Variable
from torchvision import models, datasets, transforms
from collections import OrderedDict
from PIL import Image

def transform_load_data(base_dir):
    '''
    [Transforms and Loads data]

    Transforms the test, valid and traing data
    Loads test, valid and train data

    Arguments:
    base_dir {String} -- Base Directory where the train, valid and test data files are stored

    Return:
    returns the loaders for train, valid and test data sets
    '''
    #different data directories to load the data

    location = base_dir
    train_dir = location + '/train'
    valid_dir = location + '/valid'
    test_dir = location + '/test'

    #mean and standard deviations
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomRotation(60),      # Random Rotation by 60 Degrees
                                 transforms.RandomHorizontalFlip(),   # Horizontal Flip
                                 transforms.RandomResizedCrop(224),   # Center Crop the image -> 224 X 224
                                 transforms.ToTensor(),        # Convert to Tensor
                                 transforms.Normalize(mean=means, std=stds)]) # apply mean and std

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),   # Center Crop the image -> 224 X 224
                                 transforms.ToTensor(),        # Convert to Tensor
                                 transforms.Normalize(mean=means, std=stds)]) # apply mean and std

    test_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),   # Center Crop the image -> 224 X 224
                                 transforms.ToTensor(),        # Convert to Tensor
                                 transforms.Normalize(mean=means, std=stds)]) # apply mean and std

    #Load the data sets
    image_datasets = {
                'train' : datasets.ImageFolder(train_dir, transform=train_transforms),
                'validation' : datasets.ImageFolder(valid_dir, transform=validation_transforms),
                'test' : datasets.ImageFolder(test_dir, transform=test_transforms)
                }

    #Data Loader
    dataloaders = {
                'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size = 32, shuffle = True),
                'valid' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size = 32),
                'test'  : torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32)
                }
    
    print("Returning Data loaders and Image data sets")
    return dataloaders, image_datasets

def create_nn(model_type, dropout=0.25, learning_rate=0.001, hidden_layer1=512, processing='gpu'):
    '''
    [Create the nueral network]

    [Create the selected nueral network
    The classifier and optimizer is also set]
    
    Arguments:
        out_size {integer} -- Output Size of the NN
        dropout {float} -- Drop out to be used in the classifier
        learning_rate {float} -- learning rate to be used in the Optimizer

    Keyword Arguments:
        model_type {str} -- [which pretrained to be used] (default: {'vgg19'})

    Returns:
        model -- the pre trained model
        criterion - the criterion to be used for training the model
        optimizer - the optomizer to be used for training the model
    '''

    if model_type == 'vgg19':
        print("Model : vgg19")
        model = models.vgg19(pretrained=True)
    elif model_type == 'vgg13':
        print("Model : vgg13")
        model = models.vgg13(pretrained=True)
    else:
        print("Not a valid model. Using vgg19 as the default model")
        model = models.vgg19(pretrained=True)

    input_size = model.classifier[0].in_features
    hidden_layer2 = hidden_layer1 // 2
    out_size = 102

    print("Input Size : {}".format(input_size))
    print("Hidden Layer 1 : {}".format(hidden_layer1))
    print("Hidden Layer 2 : {}".format(hidden_layer2))
    print("Output Size : {}".format(out_size))

    for param in model.parameters():
        param.requires_grads = False

    classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('dropout',nn.Dropout(p=dropout)),
            ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
            ('relu2', nn.ReLU()),
            ('dropout',nn.Dropout(p=dropout)),
            ('output', nn.Linear(hidden_layer2, out_size)),
            ('softmax', nn.LogSoftmax(dim=1))
            ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= learning_rate)
    print(learning_rate)
    
    if(torch.cuda.is_available() and processing == 'gpu'):
        model.cuda()
    
    print("Returning model, criterion, optimizer, classifier")
    return model, optimizer, criterion, classifier

def validate_model(model, validation_data, criterion, processing):
    validation_start = time.time()
    total_validation_loss = 0
    accuracy = 0
    for inputs, labels in validation_data:
        if torch.cuda.is_available() and processing=='gpu':
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

        outputs = model.forward(inputs)
        validation_loss = criterion(outputs, labels)

        ps = torch.exp(outputs.data)
        
        _, predicted = ps.max(dim=1)
        
        correct_values = predicted == labels.data
        accuracy += correct_values.type_as(torch.FloatTensor()).mean()

        total_validation_loss += validation_loss.item()
        
    validation_time = time.time() - validation_start
    print("Validation loss: {:.3f}".format(total_validation_loss/len(validation_data)),
        "Accuracy: {:.3f}".format(accuracy/len(validation_data)),
        "Validation time: {:.3f} s/batch".format(validation_time/len(validation_data)))
    return()

def train_model(model, criterion, optimizer, loader, epochs = 5, print_every=25, processing='gpu'):
    '''[Train the model]

    [Train the model on the training data set]
    
    Arguments:
        model {[type]} -- [Model to be used for training]
        criterion {[type]} -- [Criterion to be used for training]
        optimizer {[type]} -- [Optimizer to be used for Training]
        loader {[type]} -- [data Loaded to be used for training]

    Keyword Arguments:
        epochs {number} -- [Number of epochs] (default: {5})
        print_every {number} -- [Print the statistics every ] (default: {25})
        processing {str} -- [GPU or CPU] (default: {'gpu'})
    '''
    model.train()
    print("Training the model started!!")
    print(optimizer)
    print(criterion)
    print(print_every)
    for step in range(epochs):
        total_loss = 0
        counter = 0
        print(f"Epoch {step+1}/{epochs}")
    
        for i, (images, labels) in enumerate(loader['train']):      
            counter += 1

            if torch.cuda.is_available() and processing=='gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
        #Forward and Backward Passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
            if counter % print_every == 0:
                print(f"Step: {counter}")
                print(f"Training loss {total_loss/counter:.3f}")
                model.eval()
                validate_model(model, loader['valid'], criterion, processing) # Validation Function
                model.train()
                
    print("Training Completed!!")
    return()

def save_checkpoint( model, classifier, optimizer, epochs, train_data, file='checkpoint.pth'):
    '''Saves the model to a check point

    Saves the model to a check point provided by the user. Default value is checkpoint.pth
    
    Arguments:
        model {[type]} -- [Trained model]
        classifier {[type]} -- [Classifier]
        optimizer {[type]} -- [Optimizer]
        train_data {[type]} -- [Training Data Set]

    Keyword Arguments:
        file {str} -- [Cehck point path] (default: {'checkpoint.pth'})
    '''
    model.class_to_idx = train_data.class_to_idx
    model_state = {
        'epoch': epochs,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': classifier,
        'class_to_idx': model.class_to_idx,
        }
    
    torch.save(model_state, file)
    
    print("Save point created!!")
    return()

def load_checkpoint(file='checkpoint.pth'):
    '''Loads the check point

    [Loads the check point]

    Keyword Arguments:
        file {str} -- [Check point path] (default: {'checkpoint.pth'})

    Returns:
        [model] -- [returns the model]
    '''
    model_state = torch.load(file, map_location=lambda storage, loc: storage)

    model = models.vgg19(pretrained=True)
    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
           
    pil_image = Image.open(image).convert("RGB")
    
    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, std)])
    pil_image = in_transforms(pil_image)
    
    return pil_image

def predict(image_path, model, topk=5, processing='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    torch.topk() --> Returns the k largest elements of the given input tensor along a given dimension.
    If dim is not given, the last dimension of the input is chosen.
    If largest is False then the k smallest elements are returned.
    https://pytorch.org/docs/master/torch.html?highlight=torch%20topk#torch.topk
    ''' 
    model.eval()
    if torch.cuda.is_available() and processing=='gpu':
        model.cuda()
    else:
        model.cpu()
    
    image = process_image(image_path) #returns a torch tensor
    image = image.unsqueeze(0) # Unsqueeze returns a new sensor. Good learning about this functionality
    
    # No gradient calculation needed
    with torch.no_grad():
        output = torch.exp(model.forward(image))
        top_probs, top_labels = torch.topk(output, topk)
        top_probs, top_labels = top_probs.data.numpy().squeeze(), top_labels.data.numpy().squeeze()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in top_labels]
    
    return top_probs, top_classes

    