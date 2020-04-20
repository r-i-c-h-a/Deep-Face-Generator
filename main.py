# DeepFakes Face Generator

# Note to evaluator: Please look into instructions.txt in the zip file
#   for more details

# You will require our datasets:
#   1. 100k Celebrities Images Dataset: [1]
#   [1] https://www.kaggle.com/greg115/celebrities-100k
#   2. CelebA Dataset: [2]
#   [2] https://www.kaggle.com/jessicali9530/celeba-dataset

# -----------------------------------------------------------------------------

# Mount Drive to access dataset
# Commenting out - if you are running this locally
""" 
from google.colab import drive
from google.colab import files
drive.mount('/content/drive')
"""

# Notebook-specific commands
# Retrieve file names of all images in the dataset
# %cd /content/drive/My Drive/face_gen_input/outer_img_celeba
# %tensorflow_version 1.x
"""
!pip install tensorboard==1.15.0
!pip install tensorflow==1.15.2
!pip install tensorflow-gpu==1.15.2
"""

# %matplotlib inline

from tensorflow.layers import dense, batch_normalization, conv2d_transpose, conv2d
from tensorflow.nn import sigmoid_cross_entropy_with_logits as loss
from tensorflow.train import AdamOptimizer as adam
from tensorflow import reduce_mean
import tensorflow as tf
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt


imageIds = glob.glob('img_align_celeba/*')
# Size of dataset
# print(len(imageIds))

# Resize the images to a 64 x 64 for uniform input
# We could also crop the images here, if required

shape = (30, 55, 150, 175)
images = []
for index in range(len(imageIds)):
    images.append(
        np.array((Image.open(imageIds[index]).crop(shape)).resize((64, 64))))

for i in range(len(images)):
    images[i] = ((images[i] - images[i].min())/(255 - images[i].min()))
    images[i] = images[i]*2-1

images = np.array(images)


def inputs(real_dim, noise_dim):
    """
    Take `real_dim` for training data, and;
    noise_dim` for noisy data.
    """
    realInputs = tf.placeholder(
        tf.float32, (None, *real_dim), name='realInput')
    noisyInputs = tf.placeholder(
        tf.float32, (None, noise_dim), name='noisyInput')
    return realInputs, noisyInputs


def plotGraphs(disLoss, genLoss):
    """
    Graph of `disLoss` against `genLoss`.
    """
    plt.style.use('seaborn')
    plt.plot(disLoss, color="blue", label="Discriminator Loss", linewidth=2)
    plt.plot(genLoss, color="red", label="Generator Loss", linewidth=2)
    plt.ylim(top=2.5)
    plt.xlim(xmax=27)
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    # plt.savefig("100k_default_params.png")
    # files.download("100k_default_params.png")
    plt.show()


# Hyper-parameters

beta1 = 0.5  # Momentum for Adam Optimizer
alpha = 0.2  # Leaky-ReLU slope
smooth = 0.9
noiseSize = 200  # Noise dimension
learningRate = 0.0002  # Learning rate for model
inputShape = (64, 64, 3)  # 64x64 - RGB (conv2D layers)

# building the graph

tf.reset_default_graph()

realInput, noisyInput = inputs(inputShape, noiseSize)

gen_noise = generator(noisyInput)

disLogitsReal = discriminator(realInput)
disLogitsFake = discriminator(gen_noise, reuse=True)

# defining losses

shape = disLogitsReal

disLossReal = reduce_mean(
    loss(logits=disLogitsReal, labels=tf.ones_like(shape*smooth)))

disLossFake = reduce_mean(
    loss(logits=disLogitsFake, labels=tf.zeros_like(shape)))

genLossTotal = reduce_mean(
    loss(logits=disLogitsFake, labels=tf.ones_like(shape*smooth)))

disLossTotal = disLossReal + disLossFake

# defining optimizers

total_vars = tf.trainable_variables()

disVars = [var for var in total_vars if var.name[0] == 'd']
genVars = [var for var in total_vars if var.name[0] == 'g']

disOpt = adam(learning_rate=learningRate, beta1=beta1).minimize(
    disLossTotal, var_list=disVars)
genOpt = adam(learning_rate=learningRate, beta1=beta1).minimize(
    genLossTotal, var_list=genVars)


def view_samples(epoch, samples, nrows, ncols, figsize=(5, 5)):
    """
    Sample output from generator.
    """

    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

    for ax, image in zip(axes.flatten(), samples[epoch]):

        ax.axis('off')
        image = ((image - image.min())*255 /
                 (image.max() - image.min())).astype(np.uint8)
        im = ax.imshow(image)

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes


batchSize = 128  # Batch size for training
epochs = 10  # No. of epochs

# No. of iterations to train all of training data
iters = len(imageIds)//batchSize
# print(iters)

if __name__ == "__main__":
    with tf.Session() as sess:
        disLoss = []
        genLoss = []
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):

            for i in range(iters-1):

                batchImages = images[i*batchSize:(i+1)*batchSize]
                batchNoise = np.random.uniform(-1,
                                               1, size=(batchSize, noiseSize))

                sess.run(disOpt, feed_dict={
                    realInput: batchImages, noisyInput: batchNoise})
                sess.run(genOpt, feed_dict={
                    realInput: batchImages, noisyInput: batchNoise})

                if i % 50 == 0:
                    print("Epoch {}/{} -->".format(e+1, epochs),
                          "Batch No {}/{}".format(i+1, iters))

            lossDis = sess.run(
                disLossTotal, {noisyInput: batchNoise, realInput: batchImages})
            lossGen = genLossTotal.eval(
                {realInput: batchImages, noisyInput: batchNoise})
            disLoss.append(lossDis)
            genLoss.append(lossGen)

            print("Epoch {}/{} -->".format(e+1, epochs), "Discriminator Loss: {:.4f} -->".format(lossDis),
                  "Generator Loss: {:.4f}".format(lossGen))

            sampleNoise = np.random.uniform(-1, 1, size=(8, noiseSize))
            genSamples = sess.run(generator(noisyInput, reuse=True, alpha=alpha),
                                  feed_dict={noisyInput: sampleNoise})

            view_samples(-1, genSamples, 2, 4, (10, 5))
            plt.show()

    plotGraphs(disLoss, genLoss)