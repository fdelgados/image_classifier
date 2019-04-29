#!/usr/bin/env python

import argparse

from workspace_utils import active_session
from model_utils import ModelUtils, Checkpoint, Phases
from image_utils import ImageUtils

parser = argparse.ArgumentParser(description='Train a network for an image classifier',
                                 usage='python train.py path/to/data [OPTIONS]')

parser.add_argument('data_directory', help='Path to data directory')
parser.add_argument('num_classes', type=int, help='Number of output classes')

parser.add_argument('-g', '--gpu',
                    default=False,
                    action='store_true',
                    help='Enables GPU support if available. Default: %(default)s')

parser.add_argument('-s', '--save-dir',
                    dest='save_dir',
                    default='.',
                    help='Directory to save checkpoints. Default: %(default)s',
                    metavar='')

parser.add_argument('-a', '--architecture',
                    dest='architecture',
                    default='vgg',
                    help='Model architecture. Default: %(default)s',
                    metavar='')

parser.add_argument('-o', '--optimizer',
                    dest='optimizer',
                    default='adam',
                    help='Model optimizer. Default: %(default)s',
                    metavar='')

parser.add_argument('-c', '--loss-function',
                    dest='loss_function',
                    default='nll',
                    help='Loss function. Default: %(default)s',
                    metavar='')

parser.add_argument('-l', '--learning-rate',
                    dest='learning_rate',
                    default=0.001,
                    type=float,
                    help='Learning rate. Default: %(default)s',
                    metavar='')

parser.add_argument('-d', '--drop-out',
                    dest='drop_out',
                    default=0.5,
                    type=float,
                    help='Drop out probability. Default: %(default)s',
                    metavar='')

parser.add_argument('-e', '--epochs',
                    dest='epochs',
                    default=10,
                    type=int,
                    help='Number of epochs. Default: %(default)s',
                    metavar='')

parser.add_argument('-p', '--print-every',
                    dest='print_every',
                    default=10,
                    type=int,
                    help='Prints out training loss, validation loss, and validation accuracy every n images. Default: '
                         '%(default)s',
                    metavar='')

args = parser.parse_args()

data_directory = args.data_directory
num_classes = args.num_classes
gpu = args.gpu
save_dir = args.save_dir
architecture = args.architecture
optimizer = args.optimizer
loss_function = args.loss_function
learning_rate = args.learning_rate
drop_out = args.drop_out
epochs = args.epochs
print_every = args.print_every

image_utils = ImageUtils(data_directory)
model_utils = ModelUtils(gpu)

print('\nBuilding model...\n')

model = model_utils.build_model(architecture, num_classes, dropout_prob=drop_out)
data_loaders = image_utils.create_data_loaders()

optimizer = model_utils.create_optimizer(model, optimizer, learning_rate=learning_rate)
criterion = model_utils.loss_function(loss_function)

print('\nTraining network...\n')
with active_session():
    trained_model = model_utils.train_network(model,
                                              data_loaders,
                                              optimizer,
                                              criterion,
                                              num_epochs=epochs,
                                              print_every=print_every)

print('\nEnd of network training.')

print('\nChecking network accuracy...\n')
accuracy = model_utils.test_network(trained_model, data_loaders[Phases.TEST_PHASE])

if accuracy * 100 > 70:
    print('\nSaving checkpoint')
    checkpoint = Checkpoint(trained_model,
                            architecture,
                            num_classes,
                            drop_out,
                            image_utils.train_dataset().class_to_idx)

    path = model_utils.save_checkpoint(checkpoint, save_dir)

    print('\nCheckpoint saved in path {}'.format(path))
else:
    print('The network accuracy is too low. Train the network again modifying parameters')
