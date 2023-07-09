import torch
from torch.autograd import Variable
import argparse
import numpy as np
import json
from PIL import Image



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', dest='checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', type=int, dest='top_k', default='5')
    parser.add_argument('--filepath', dest='filepath', default="flowers/test/1/image_06760.jpg")
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True)
    return parser.parse_args()

"""
    Arg Parse Parameters
    - checkpoint is where the saved model is stored 
    - top_k: number of top probabilities 
    - filepath: path of image to check
    - category_names: names dict to check predictions
    - gpu: set to gpu (cuda) or cpu
"""

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names


def load_checkpoint(filepath): 
    saved_model = torch.load(filepath)
    model = saved_model['arch']
    return model


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
   """
    # resize to 256x256
    resized = image.resize((256, 256))
    height, width = resized.size

    # crop the center of the image
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image_crop = resized.crop((left, top, right, bottom))

    # normalize
    image_norm = np.array(image_crop) / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_norm = (image_norm - mean) / std

    # reorder for PIL
    image_trans = image_norm.transpose((2, 0, 1))
    return image_trans


def predict(image_path, model, topk, gpu):

    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    # turn off dropout
    with torch.no_grad():
        model.eval()

    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)

    inputs = Variable(tensor.float().cuda())
    inputs = inputs.unsqueeze(0)
    output = model.forward(inputs)

    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu()
    classes = ps[1].cpu()
    class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()

    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])

    return probabilities.numpy()[0], mapped_classes
