import os
import numpy as np
import torch
from torch import nn, optim
from torchvision import models


class Phases:
    TRAIN_PHASE = 'train'
    VALIDATION_PHASE = 'validation'
    TEST_PHASE = 'test'

    def __call__(self):
        return [self.TRAIN_PHASE, self.VALIDATION_PHASE, self.TEST_PHASE]


class Checkpoint:
    def __init__(self, model, architecture, num_classes, drop_out, class_to_idx):
        model.class_to_idx = class_to_idx

        self.architecture = architecture
        self.output_size = num_classes
        self.dropout_prob = drop_out
        self.model_state_dict = model.state_dict()
        self.class_to_idx = model.class_to_idx

    def to_dict(self):
        return {'architecture': self.architecture,
                'output_size': self.output_size,
                'dropout_prob': self.dropout_prob,
                'model_state_dict': self.model_state_dict,
                'class_to_idx': self.class_to_idx}


class ModelUtils:
    def __init__(self, gpu_support=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu_support else 'cpu')

    def build_model(self, architecture, num_classes, dropout_prob=0.2):
        model = None
        if architecture == 'vgg':
            ''' vgg16 Layers
            (classifier): Sequential(
                (0): Linear(in_features=25088, out_features=4096, bias=True)
                (1): ReLU(inplace)
                (2): Dropout(p=0.5)
                (3): Linear(in_features=4096, out_features=4096, bias=True)
                (4): ReLU(inplace)
                (5): Dropout(p=0.5)
                (6): Linear(in_features=4096, out_features=1000, bias=True)
            )
            '''

            model = models.vgg16(pretrained=True)
            model.name = 'VGG16'

            self.freeze_parameters(model)

            num_features = model.classifier[6].in_features
            model.classifier[2] = nn.Dropout(dropout_prob)
            model.classifier[5] = nn.Dropout(dropout_prob)
            model.classifier[6] = nn.Linear(num_features, num_classes)

            model.classifier.add_module('out', nn.LogSoftmax(dim=1))

        elif architecture == 'densenet':
            ''' densenet121 layers
            (classifier): Linear(in_features=1024, out_features=1000, bias=True)
            '''

            model = models.densenet121(pretrained=True)
            model.name = 'DenseNet121'

            self.freeze_parameters(model)

            num_features = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Linear(num_features, 512),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_prob),
                                             nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Dropout(dropout_prob),
                                             nn.Linear(256, num_classes))
            model.classifier.add_module('out', nn.LogSoftmax(dim=1))

        elif architecture == 'resnet':
            ''' resnet18 layers
            (fc): Linear(in_features=512, out_features=1000, bias=True)
            '''

            model = models.resnet18(pretrained=True)
            model.name = 'ResNet18'

            self.freeze_parameters(model)

            num_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(num_features, 256),
                                     nn.ReLU(),
                                     nn.Dropout(dropout_prob),
                                     nn.Linear(256, num_classes))
            model.fc.add_module('out', nn.LogSoftmax(dim=1))

        else:
            raise ValueError('{} is not a valid architecture'.format(architecture))

        print('\nNetwork architecture: {}\n'.format(model.name))

        return model

    def freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def create_optimizer(self, model, optimizer_name='sgd', learning_rate=0.003):
        try:
            classifier = model.fc
        except AttributeError:
            classifier = model.classifier

        optimizer = None
        params = []
        for _, param in classifier.named_parameters():
            if param.requires_grad:
                params.append(param)

        if optimizer_name == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate)
        else:
            raise ValueError('{} is not a valid optimizer name'.format(optimizer_name))

        print('Optimizer: {}'.format(optimizer.__class__.__name__))

        return optimizer

    def loss_function(self, func_name='nll'):
        loss_function = None

        if func_name == 'nll':
            loss_function = nn.NLLLoss()
        elif func_name == 'cross_entropy':
            loss_function = nn.CrossEntropyLoss()
        else:
            raise ValueError('{} is not a valid loss function name'.format(func_name))

        print('Loss function: {}'.format(loss_function.__class__.__name__))

        return loss_function

    def save_checkpoint(self, checkpoint, save_dir='.'):
        if save_dir != '.':
            os.mkdir(save_dir)

        checkpoint_path = '{}/checkpoint.pth'.format(save_dir)

        torch.save(checkpoint.to_dict(), checkpoint_path)

        return checkpoint_path

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)

        model = self.build_model(checkpoint['architecture'],
                                 checkpoint['output_size'],
                                 checkpoint['dropout_prob'],
                                 silent=True)

        model.class_to_idx = checkpoint['class_to_idx']

        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def train_network(self, model, dataloaders, optimizer, criterion, num_epochs=25, print_every=5):
        steps = 0
        running_loss = 0

        model.to(self.device)

        for epoch in range(num_epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 60)

            model.train()
            for images, labels in dataloaders[Phases.TRAIN_PHASE]:
                steps += 1
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loader = dataloaders[Phases.VALIDATION_PHASE]
                    test_loss, accuracy = self.validate_network(model, validation_loader, criterion)

                    train_loss = running_loss / print_every
                    val_loss = test_loss / len(validation_loader)
                    val_acc = accuracy / len(validation_loader)

                    print('Train loss: {:.3f} | Validation loss: {:.3f} | Accuracy: {:.3f}'.format(train_loss,
                                                                                                   val_loss,
                                                                                                   val_acc))
                    running_loss = 0

        return model

    def validate_network(self, model, dataloader, criterion):
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()
                ps = torch.exp(logps)
                _, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        model.train()

        return test_loss, accuracy

    def test_network(self, model, testloader):
        model.to(self.device)
        model.eval()

        accuracy = 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = model.forward(images)

                ps = torch.exp(output)
                _, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        percent_accuracy = (accuracy * 100) / len(testloader)
        print('Network ccuracy: {:.2f}%'.format(percent_accuracy))

        return percent_accuracy

    def predict(self, image, model, topk=5):
        """
        Predict the class (or classes) of an image using a trained deep learning model.
        """

        img_tensor = torch.from_numpy(image).type(torch.FloatTensor)
        img_tensor.unsqueeze_(0)

        model.to(self.device)
        model.eval()

        with torch.no_grad():
            output = model.forward(img_tensor.to(self.device))

        ps = torch.exp(output)

        top_preds = ps.topk(topk)
        probs = np.array(top_preds[0].cpu())[0]
        top_classes = np.array(top_preds[1].cpu())[0]

        idx_to_class = {cls: idx for idx, cls in model.class_to_idx.items()}
        classes = [idx_to_class[cls] for cls in top_classes]

        return probs, classes
