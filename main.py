import torch
from torchvision import models, transforms
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread

from tqdm import tqdm


class ActivationMaximizationVis():

    def __init__(self, model, epochs, cnn_layer, cnn_filter):
        self.model = model
        self.model.eval()  # set model to evaluation mode
        self.epochs = epochs
        self.cnn_layer = cnn_layer
        self.cnn_filter = cnn_filter
        self.conv_output = 0  # initialize the output of the model for loss

        if not os.path.exists('activ_max_imgs'):
            os.makedirs('activ_max_imgs')

    def hook_cnn_layer(self):
        """ Initiates a forward hook function to save the activations
            from the selected cnn layer.
        """

        def hook_fn(module, ten_in, ten_out):  # hook functions require 3 arguments, module, in, out
            self.conv_output = ten_out[0, self.cnn_filter]
            self.num_filters = ten_out.shape[1]  # saving the number of filters in that layer

        self.model[self.cnn_layer].register_forward_hook(hook_fn)

    def vis_cnn_layer(self):
        """ Method to visualize selected filter (activation map) from
            a CNN layer. Creates a random image as input.
        """
        # initiate hook function
        self.hook_cnn_layer()
        noisy_img = np.random.randint(125, 190, (224, 224, 3), dtype='uint8')

        # add dimension and activate requires_grad on tensor
        processed_image = process_image(noisy_img).unsqueeze_(0).requires_grad_()
        optimizer = torch.optim.Adam([processed_image], lr=0.1, weight_decay=1e-6)

        for e in tqdm(range(1, self.epochs)):
            optimizer.zero_grad()  # zero out gradients
            x = processed_image

            # iterate through each layer of the model
            for idx, layer in enumerate(self.model):
                # pass processed image through each layer
                x = layer(x)
                # Stop evaluating and building the computational graph when desired layer is reached
                if idx == self.cnn_layer:
                    break

            loss = -torch.mean(self.conv_output)  # The loss is defined as the negative mean of the activation of the specified filter in the desired layer.
            # This is because gradient ascent seeks to maximize the activation.

            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights
            self.layer_img = rebuild_image(processed_image)  # reconstruct image

            print('Epoch {}/{} --> Loss {:.3f}'.format(e + 1, self.epochs, loss.data.numpy()))

        img_path = 'activ_max_imgs/am_vis_l_' + str(self.cnn_layer) + \
                   '_f' + str(self.cnn_filter) + '_iter' + str(e + 1) + '.jpg'
        save_image(self.layer_img, img_path)


def process_image(image, dim=224):
    """ Scales, crops (224 x 224 px), and normalizes a PIL image for a
        Pytorch model. Accepts both a jpg or radom nois np.ndarray. Converts
        np.ndarray to a PIL image with shape (3, 224, 224).
    """

    if isinstance(image, (np.ndarray)):
        im = Image.fromarray(image)
    else:
        im = Image.open(image)

    # resize image
    width, height = im.size
    if width > height:
        ratio = width / height
        im.thumbnail((ratio * 256, 256))
    elif height > width:
        ratio = height / width
        im.thumbnail((256, ratio * 256))
    new_width, new_height = im.size

    # crop image around center
    left = (new_width - dim) / 2
    top = (new_height - dim) / 2
    right = (new_width + dim) / 2
    bottom = (new_height + dim) / 2
    im = im.crop((left, top, right, bottom))

    # convert to a np.array and divide by the color channel (int max)
    np_image = np.array(im) / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = (np_image - mean) / std
    # convert to a Tensor - reorder color channel so it is first. Torch requirement
    image = torch.FloatTensor(image.transpose(2, 0, 1))
    return image


def rebuild_image(tensor):
    """ Rebuilds a Pytorch Tensor with dimensions (1, 3, w, h) and converts
        it to the necessary format for visualization. Reverses the normalization
        step using the mean and std from the ImageNet dataset.
    """
    np_image = tensor.detach().numpy()  # convert tensor to nparray
    np_image = np_image.squeeze(0)  # reduce size of tensor
    np_image = np_image.transpose(1, 2, 0)  # reorder color channel
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np_image * std + mean)  #

    return image


def save_image(img, path):
    """ Displays and saves the processed image from the
        given layer/filter number.
    """
    plt.figure(figsize=[2, 2])
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    plt.savefig(path, dpi=150)
    plt.close()


def extract_layer_number(image_name):
    """ Returns the layer number of the picture using a regular expression
    """
    import re
    match = re.search(r'_l_(\d+)_', image_name)
    if match:
        return int(match.group(1))


def plot_all_images_in_folder(folder_path):
    """ Iterates over the saved images in a given folder and plots it in a single subplot.
        """
    images = [f for f in os.listdir(folder_path) if f.endswith(".jpg")]
    images.sort(key=extract_layer_number)  # Sort images based on layer numbers
    num_images = len(images)

    # Calculate the grid size based on the number of images
    num_rows = (num_images - 1) // 5 + 1
    num_cols = min(num_images, 5)

    fig, axis = plt.subplots(num_rows, num_cols, figsize=(15, 10))  # Adjusted figsize

    for ax in axis.flatten():
        ax.axis('off')

    for i, image_name in enumerate(images):
        img = imread(os.path.join(folder_path, image_name))
        row = i // num_cols  # Row index
        col = i % num_cols   # Column index
        axis[row, col].imshow(img)
        axis[row, col].set_title(f"Layer {extract_layer_number(image_name)}")
        axis[row, col].axis('off')  # Turn off axis labels
    plt.tight_layout()

    plt.show()


def get_ma_from_layers(cnn_layer):
    for i in range(1, cnn_layer, 2):
        out_layer = ActivationMaximizationVis(model, epochs, i, cnn_filter)
        out_layer.vis_cnn_layer()


cnn_layer = 27
cnn_filter = 28
epochs = 51

model = models.vgg16(pretrained=True).features
get_ma_from_layers(cnn_layer)
plot_all_images_in_folder('activ_max_imgs')
