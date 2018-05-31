# Imports here
import torch
import sys
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import OrderedDict
  

if __name__ == '__main__':
    '''
    This script trains a model based on the code in the dataset given
    as an argument when running the script.
    '''
    # Get arguements
    args = sys.argv[:]
    
    if "--save_dir" in args:
        save_dir = args[args.index("--save_dir") + 1]
    else:
        save_dir = False
        
    if "--arch" in args:
        arch = args[args.index("--arch") + 1].lower()
                    
        if arch == 'alexnet': model = models.alexnet(pretrained=True)
        elif arch == 'vgg11': model = models.vgg11(pretrained=True)
        elif arch == 'vgg13': model = models.vgg13(pretrained=True)
        elif arch == 'vgg16': model = models.vgg16(pretrained=True)
        elif arch == 'vgg19': model = models.vgg19(pretrained=True)
        elif arch == 'vgg11bn': model = models.vgg11_bn(pretrained=True)
        elif arch == 'vgg13bn': model = models.vgg13_bn(pretrained=True)
        elif arch == 'vgg16bn': model = models.vgg16_bn(pretrained=True)
        elif arch == 'vgg19bn': model = models.vgg19_bn(pretrained=True)
        elif arch == 'resnet18': model = models.resnet18(pretrained=True)
        elif arch == 'resnet34': model = models.resnet34(pretrained=True)
        elif arch == 'resnet50': model = models.resnet50(pretrained=True)
        elif arch == 'resnet101':	model = models.resnet101(pretrained=True)
        elif arch == 'resnet152':	model = models.resnet152(pretrained=True)
        elif arch == 'squeezenet': model = models.squeezenet1_0(pretrained=True)
        elif arch == 'queezenet': model = models.squeezenet1_1(pretrained=True)
        elif arch == 'densenet121': model = models.densenet121(pretrained=True)
        elif arch == 'densenet169': model = models.densenet169(pretrained=True)
        elif arch == 'densenet201': model = models.densenet201(pretrained=True)
        elif arch == 'densenet161': model = models.densenet161(pretrained=True)
        elif arch == 'inception':	model = models.inception_v3(pretrained=True)
        else: 
            print('Model not recongized')
            sys.exit()
    else:
        model = models.densenet121(pretrained=True)
        
    if "--learning_rate" in args:
        learning_rate = args[args.index("--learning_rate") + 1]
    else:
        learning_rate = 0.001
        
    if "--hidden_units" in args:
        hidden_layers = args[args.index("--hidden_units") + 1]
    else:
        hidden_layers = 490
        
    if "--epochs" in args:
        epochs = args[args.index("--epochs") + 1]
    else:
        epochs = 3
        
    if "--gpu" in args:
        gpu = True
    else:
        gpu = False
        
    # Create a dataset on input argument on command line.
    data_dir = str(sys.argv[1])
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    new_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir,
                                          transform=data_transforms)
    
    valid_datasets = datasets.ImageFolder(valid_dir,
                                          transform=new_data_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loaders = DataLoader(train_datasets,
                               batch_size=32,
                               shuffle=True)
    
    valid_loaders = DataLoader(valid_datasets,
                               batch_size=32,
                               shuffle=True)
    
    # TODO: Build and train your network
    # Classifier params
    params = OrderedDict([('fc1', nn.Linear(1024, hidden_layers)),
                          ('relu', nn.ReLU()), 
                          ('fc2', nn.Linear(hidden_layers, int(hidden_layers/5))),
                          ('drop', nn.Dropout(p=0.33)),
                          ('output', nn.LogSoftmax(dim=1))])
    # Feature parameters
    for x in model.parameters():
        x.requires_grad = False
    
    model.classifier = nn.Sequential(params)
    
    # CUDA, GPU if available else CPU
    model.cuda() if gpu else model.cpu()
    
    # Optimize model
    steps = 0
    r_loss = 0.0
    n_print = 10
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    for e in range(epochs):
        # Start epoch runs
        print('Starting training...')
        print('Training run {}...'.format((e + 1)))
        
        model.train()
        
        # Iterate through images/labels
        for images, labels in iter(train_loaders):
            steps += 1
            optimizer.zero_grad()
    
            # Flatten image
            #images.resize_(images.size()[0], 50176)
            
            inputs = Variable(images)
            train_labels = Variable(labels)
            
            # CUDA, convert to CUDA objects if available
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                train_labels = train_labels.cuda()
            
            # Calculate loss
            output = model.forward(inputs)
            loss = criterion(output, train_labels)
            loss.backward()
            r_loss += loss.data[0]
            
                
            # Save Optimization
            optimizer.step()
            
            # Validation Runs
            if steps % n_print == 0:
                model.eval()
                accuracy = 0.0
                valid_loss = 0.0
                
                for i, (images, labels) in enumerate(valid_loaders):
                    inputs = Variable(images, volatile=True)
                    valid_labels = Variable(labels, volatile=True)
                    
                    output = model.forward(inputs)
                    valid_loss += criterion(output, labels).data[0]
                    
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                print("Training Loss: {:.3f}.. ".format(loss))
                print("Validation Loss: {:.3f}..".format(valid_loss/len(valid_loaders)))
                print("Validation Accuracy: {:.3f}..\n".format(accuracy/len(valid_loaders)))
                    
                r_loss = 0.0
                model.train()
                
    # TODO: Save the checkpoint
    if save_dir: 
        check_point_path = save_dir + '\checkpoint.pth'
    else: 
        check_point_path = 'checkpoint.pth'

    torch.save(model.state_dict(), check_point_path)
    model.class_to_idx = train_datasets.class_to_idx
    
    
                