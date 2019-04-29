#!/usr/bin/env python

import argparse
from model_utils import ModelUtils
from image_utils import categories_map, process_image


def print_predict_table(probs, categories, actual):
    """
    Prints a formatted table with prediction results
    """
    cat_names = [cat_to_name[str(cat)] for cat in categories]
    percent = [prob * 100 for prob in probs]
    probabilities = dict(zip(cat_names, percent))

    print('   {0:25} PROBABILITY (%)'.format('FLOWER'))
    print('-' * 45)

    pred_num = 1
    for category, probability in probabilities.items():
        if actual == category:
            category = '{} (*)'.format(category)
        print('{:>2} {:25}{:>15.2f}'.format(pred_num, category.title(), probability))
        pred_num += 1

    print('')


parser = argparse.ArgumentParser(description='Predict image category',
                                 usage='python predict.py path/to/image path/to/checkpoint_file [OPTIONS]')

parser.add_argument('input', help='Image path')
parser.add_argument('checkpoint', help='Checkpoint file. This will be used to rebuild the model.')

parser.add_argument('-k', '--top-k',
                    dest='top_k',
                    default=5,
                    type=int,
                    help='The top k most probable classes to be displayed. Default: %(default)s',
                    metavar='')

parser.add_argument('-c', '--category-names',
                    dest='category_names',
                    default='cat_to_name.json',
                    help='A JSON file that maps class integer encoding to actual flower names. Default: %(default)s',
                    metavar='')

parser.add_argument('-g', '--gpu',
                    default=False,
                    action='store_true',
                    help='Enables GPU support if available. Default: %(default)s')

args = parser.parse_args()

image_path = args.input
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

cat_to_name = categories_map(category_names)

cat_name = cat_to_name[image_path.split('/')[2]]

print('\nPredicting flower name...')
print('\nActual Flower Name: {}\n'.format(cat_name.title()))

model_utils = ModelUtils(gpu)
model = model_utils.load_checkpoint(checkpoint)

image = process_image(image_path)

probs, categories = model_utils.predict(image,
                                        model,
                                        topk=top_k)

print_predict_table(probs, categories, cat_name)
