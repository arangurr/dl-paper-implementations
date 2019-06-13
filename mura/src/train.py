import argparse
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vis.visualization as vis

from keras import backend as K
from keras import optimizers
from keras.applications import DenseNet169
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import Sequence

from skimage import transform
from sklearn.metrics import roc_auc_score, precision_score, recall_score, log_loss, accuracy_score
from sklearn.model_selection import train_test_split


# Prepares image for input into the model
def transform_image(image, augment):
    image *= 1. / 255

    if augment:
        if random.randint(0, 1):
            # Lateral Flip
            image = np.fliplr(image)

        angle = random.randint(-30, 30)
        image = transform.rotate(image, angle)

    return image


# Keras image generator
class MuraGenerator(Sequence):
    def __init__(self, paths_studies, batch_size, weights, augment=False):
        self.bs = batch_size
        self.paths_studies = paths_studies
        self.augment = augment
        self.weights = weights

    def __len__(self):
        return len(self.paths_studies) // self.bs

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        w_batch = []

        paths = self.paths_studies[idx * self.bs: (idx + 1) * self.bs]
        for path in paths:
            section = path[0][16:23]
            img_paths = glob(str(path[0]) + '*')
            # Every batch contains bs studies, each one with at least one image.
            for img_path in img_paths:
                img = image.load_img(img_path, color_mode='rgb',
                                     target_size=(320, 320))

                img = image.img_to_array(img)

                img = transform_image(img, self.augment)

                x_batch.append(img)
                y_batch.append(int(path[1]))
                # The weights for each image, to solve class imbalance.
                w_batch.append(self.weights[section][int(path[1])])

        return [np.asarray(x_batch), np.asarray(y_batch), np.asarray(w_batch)]


def generate_model(stage):
    K.clear_session()
    # The denseNet, include_top automaticly eliminates the last layers
    base_model = DenseNet169(include_top=False,
                             input_shape=(320, 320, 3),
                             weights='imagenet')

    x = base_model.output
    # The last layer is replace by a pooling and a dense
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid', name='predictions')(x)

    # Model using the functional API
    model = Model(inputs=base_model.inputs, outputs=x)

    sgd = optimizers.SGD(lr=1e-4)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=sgd,
                  metrics=['binary_accuracy'],
                  loss='binary_crossentropy')

    # If in stage 2 or more, load the weights from the input model
    if stage >= 2:
        if args.model_path:
            model.load_weights(args.model_path, by_name=True)
            print("Loaded from: ", args.model_path)

        # Set layers to be trainable.
        if stage == 2:
            set_trainable = False
            for layer in base_model.layers:
                # if "block12" in layer.name:  # what block do we want to start unfreezing
                set_trainable = True
                if set_trainable:
                    layer.trainable = True
                else:
                    layer.trainable = False

    # Recompile with the loaded weights.
    model.compile(optimizer=sgd,
                  metrics=['binary_accuracy'],
                  loss='binary_crossentropy')

    if args.print_summary == True:
        model.summary()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MURA image classification")
    parser.add_argument('-r', '--resume', action='store_true', default='False',
                        help='Resume training from last saved model')
    parser.add_argument('-s', '--stage', default=0, type=int,
                        help='Set stage of training: '
                             '0-train only dense layer'
                             '1-train only dense layer with image augmentation.'
                             '2-train dense with augmentation and last conv block.'
                             '3-testing, report all metric of the test data.'
                             '4-evaluate a single client, indicated with -c, plot image and CAM.')
    parser.add_argument('--train_path', default='MURA-v1.1/train_labeled_studies.csv',
                        help='Path to train_labeled_studies.csv')
    parser.add_argument('--train_images', default='MURA-v1.1/train_image_paths.csv',
                        help='Path to train_image_paths.csv')
    parser.add_argument('--test_path', default='MURA-v1.1/valid_labeled_studies.csv',
                        help='Path to valid_labeled_studies.csv')
    parser.add_argument('--test_images', default='MURA-v1.1/valid_image_paths.csv',
                        help='Path to valid_image_paths.csv')
    parser.add_argument('--model_path',
                        help='Path to a model to resume or proceed with transfer learning')
    parser.add_argument('-c', '--client', default=0,
                        help='Client to evaluate')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='number of epochs to train')
    parser.add_argument('--section', default='XR_WRIS', type=str,
                        help='XR_SHOU, XR_HUME, XR_FORE, XR_HAND, XR_ELBO, XR_FING, XR_WRIS')
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--print_summary', action='store_true', default='False',
                        help='print model\'s summary')
    args = parser.parse_args()

    starting_epoch = 0

    # Read the train path and filter for the given section
    studies_path = np.asarray(pd.read_csv(args.train_path, delimiter=',', header=None))
    if args.section is not None:
        studies_path = [i for i in studies_path if args.section in i[0]]

    model = generate_model(int(args.stage))

    # On resume, read the epoch to start at.
    if args.resume is True:
        starting_epoch = int(args.model_path[25:28])

    weights = {
        "XR_SHOU": [0, 0],
        "XR_HUME": [0, 0],
        "XR_FORE": [0, 0],
        "XR_HAND": [0, 0],
        "XR_ELBO": [0, 0],
        "XR_FING": [0, 0],
        "XR_WRIS": [0, 0]
    }

    # Go through the images and obtain the class weights
    paths_imgs = np.loadtxt(args.train_images, dtype='str')
    for path in paths_imgs:
        section = path[16:23]
        if "positive" in path:
            weights[section][1] += 1
        elif "negative" in path:
            weights[section][0] += 1

    # Normalize the weights.
    for section in weights:
        weights[section] = weights[section] / np.sum(weights[section])

    if args.stage < 3:
        # Split into train and validation with a constant seed to maintain state on resume.
        train_paths, val_paths = train_test_split(studies_path, random_state=0)

        train_generator = MuraGenerator(train_paths, batch_size=args.batch_size, weights=weights,
                                        augment=True if args.stage > 0 else False)
        val_generator = MuraGenerator(val_paths, batch_size=args.batch_size, weights=weights)

        if not os.path.isdir('logs'):
            os.mkdir('logs')
        if not os.path.isdir('models'):
            os.mkdir('models')

        # Both names from the checkpointer and the logger will modified the name of the files
        # This will avoid deleting data on new experiments. If resuming, the logger appends data.
        checkpoint_path = 'models/stage-{}-'.format(args.stage) + '-model-e{epoch:03d}.hdf5'
        csvlogger = CSVLogger('logs/stage-{}.log'.format(args.stage), append=args.resume)
        checkpointer = ModelCheckpoint(checkpoint_path, save_best_only=False,
                                       save_weights_only=True)

        model.fit_generator(train_generator,
                            callbacks=[csvlogger, checkpointer],
                            epochs=args.epochs,
                            initial_epoch=starting_epoch,
                            validation_data=val_generator)

    # Obtain the metrics for the test dataset
    elif args.stage == 3:
        # Read the test set and filter for the selected section
        studies_path = np.asarray(pd.read_csv(args.test_path, delimiter=',', header=None))
        if args.section is not None:
            studies_path = [path for path in studies_path if args.section in path[0]]

        y_pred = []
        y_true = []
        y_pred_bin = []
        sample_w = []
        # FOr each study, read the images and process them
        for study in studies_path:
            section = study[0][16:23]
            img_paths = glob(str(study[0]) + '*')
            images = []
            for img_path in img_paths:
                img = image.load_img(img_path, color_mode='rgb',
                                     target_size=(320, 320))

                img = image.img_to_array(img)

                img = transform_image(img, False)
                images.append(img)

            # Predict values, obtain the mean of the probabilities of the images as y_pred.
            images = np.asarray(images)
            results = model.predict_on_batch(images)
            y_pred.append(np.mean(results))

            y_true.append(int(study[1]))
            sample_w.append(weights[section][int(study[1])])

        y_pred_bin = [0 if i <= 0.5 else 1 for i in y_pred]

        # Obtain the validation metrics and print them
        print("Scores: ")
        print("Number of studies: ", len(y_true))
        print("\tLoss: ", log_loss(y_true, y_pred))
        print("\tAccuracy: ", accuracy_score(y_true, y_pred_bin))
        print("\tRecall: ", recall_score(y_true, y_pred_bin))
        print("\tPrecision: ", precision_score(y_true, y_pred_bin))
        print("\tAUC: ", roc_auc_score(y_true, y_pred))

    # For a given client study, show model output (on console) and the CAMs of each image.
    elif args.stage == 4:
        studies_path = np.asarray(pd.read_csv(args.test_path, delimiter=',', header=None))

        # Search test set for the specific client
        client = args.client.zfill(5)
        studies_path = [path for path in studies_path if str(client) in path[0]]

        y_pred = []
        images = []
        show_ims = []

        # Read and process all the images
        img_paths = glob(str(studies_path[0][0]) + '*')
        for path in img_paths:
            img = image.load_img(path, color_mode='rgb',
                                 target_size=(320, 320))
            show_ims.append(img)
            img = image.img_to_array(img)

            img /= 255.
            images.append(img)

        images = np.asarray(images)
        # Generate predictions
        results = model.predict_on_batch(images)

        # Save the mean of the probabilities and the true result for printing
        y_pred = np.mean(results)
        y_true = int(studies_path[0][1])

        print("Results: ")
        print("\ty_true: ", y_true)
        print("\ty_pred: ", y_pred)

        for i in range(len(images)):
            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(images[i].squeeze())
            plt.axis('off')
            plt.title('Input Image')

            plt.subplot(1, 2, 2)
            plt.imshow(images[i].squeeze(), 'gray', alpha=0.7)
            # Obtain the cams, of layer -1 (the last one) for the input image[i]
            heat_map = vis.visualize_cam(model, -1, None, images[i])
            plt.imshow(heat_map, alpha=0.3)
            plt.axis('off')
            plt.title('Activation Map')

            plt.tight_layout()
            plt.savefig('logs/map' + str(i) + '.png', dpi=200)
