import os
import random


def random_image(image_path, max_class_id):
    random.seed(1)

    category = random.randint(1, max_class_id)

    return random.choice(os.listdir('{}/{}'.format(image_path, category)))
