import numpy as np
from tqdm import trange, tqdm
import glob
import h5py

from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing import image

import random

from models import components, mae_loss, mse_loss, color_loss


# Avoid crash on non-X linux sessions (tipically servers) when plotting images
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Images size
w = 256
h = 104
colors_number = 5

# Cyclic consistency factor

lmda = 10

# Optimizer parameters

lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
epsilon = 1e-08

# Setting image format as (channels, height, width)
K.set_image_dim_ordering('th')

disc_a_history = []
disc_b_history = []

gen_a2b_history = {'bc':[], 'mae':[]} 
gen_b2a_history = {'bc':[], 'mae':[]}

gen_b2a_history_new = []
gen_a2b_history_new = []
cycle_history = []
# Data loading

def extract_colors(data):
    colors_data = np.empty((data.shape[0], colors_number, 3), dtype=np.float32)
    j = 0
    print("Extracting colors information...")
    for image in tqdm(data):
        k = 0
        colors = np.empty((colors_number, 3), dtype=np.float32)

        temp = image.transpose(1,2,0)

        # extract dominant colors from the colors stripe at the bottom of the image
        for pixel in temp[temp.shape[2] - 5]:
            found = False
            for element in colors:
                if np.array_equal(pixel, element):
                    found = True
                    break
            if not found:
                colors[k] = pixel
                k += 1

        # normalize the colors vector
        for i in range(colors_number):
            colors[i] = (colors[i].astype(np.float32) - 127.5) / 127.5

        # add it to the dataset
        colors_data[j] = colors
        j += 1

    return colors_data

def loadImage(path, h, w):
    
    '''Load single image from specified path'''
    img = image.load_img(path)
    img = img.resize((w,h))
    x = image.img_to_array(img)
    return x


def loadImagesFromDataset(h, w, dataset, use_hdf5=False):

    '''Return a tuple (trainA, trainB, testA, testB) 
    containing numpy arrays populated from the
     test and train set for each part of the cGAN'''

    if (use_hdf5):
        path="./datasets/processed/"+dataset+"_data.h5"
        data = []
        print('\n', '-' * 15, 'Loading data from dataset', dataset, '-' * 15)
        with h5py.File(path, "r") as hf:
            for set_name in tqdm(["trainA_data", "trainB_data", "testA_data", "testB_data"]):
                if dataset == "nike2adidas" or ("adiedges" in dataset) and "train" in set_name:
                    data.append(hf[set_name][:1000].astype(np.float32))
                else:
                    data.append(hf[set_name][:].astype(np.float32))

        return (set_data for set_data in data)

    else:
        path = "./datasets/"+dataset
        print(path)
        train_a = glob.glob(path + "/trainA/*.png")
        train_b = glob.glob(path + "/trainB/*.png")
        test_a = glob.glob(path + "/testA/*.png")
        test_b = glob.glob(path + "/testB/*.png")

        print("Import trainA")
        if dataset == "nike2adidas" or ("adiedges" in dataset):
            tr_a = np.array([loadImage(p, h, w) for p in tqdm(train_a[:1000])])
        else:
            tr_a = np.array([loadImage(p, h, w) for p in tqdm(train_a)])

        print("Import trainB")
        if dataset == "nike2adidas" or ("adiedges" in dataset):
            tr_b = np.array([loadImage(p, h, w) for p in tqdm(train_b[:1000])])
        else:
            tr_b = np.array([loadImage(p, h, w) for p in tqdm(train_b)])

        print("Import testA")
        ts_a = np.array([loadImage(p, h, w) for p in tqdm(test_a)])

        print("Import testB")
        ts_b = np.array([loadImage(p, h, w) for p in tqdm(test_b)])

    # Extract dominant colors data from the picture if necessary
    if dataset == "adiedges4":
        colors_tr_a = extract_colors(tr_a)
        colors_tr_b = extract_colors(tr_b)
        return tr_a, tr_b, ts_a, ts_b, colors_tr_a, colors_tr_b
    return tr_a, tr_b, ts_a, ts_b
    


# Create a wall of generated images

def plotGeneratedImages(epoch, set_a, set_b, generator_a2b, generator_b2a, examples=6):
    
    true_batch_a = set_a[np.random.randint(0, set_a.shape[0], size=examples)]
    true_batch_b = set_b[np.random.randint(0, set_b.shape[0], size=examples)]

    # Get fake and cyclic images
    generated_a2b = generator_a2b.predict(true_batch_a)
    cycle_a = generator_b2a.predict(generated_a2b)
    generated_b2a = generator_b2a.predict(true_batch_b)
    cycle_b = generator_a2b.predict(generated_b2a)
    
    k = 0

    # Allocate figure
    plt.figure(figsize=(w/10, h/10))

    for output in [true_batch_a, generated_a2b, cycle_a, true_batch_b, generated_b2a, cycle_b]:
        output = (output+1.0)/2.0
        for i in range(output.shape[0]):
            plt.subplot(examples, examples, k*examples +(i + 1))
            img = output[i].transpose(1, 2, 0)  # Using (ch, h, w) scheme needs rearranging for plt to (h, w, ch)
            #print(img.shape)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout()
        k += 1
    plt.savefig("images/epoch"+str(epoch)+".png")
    plt.close()


# Plot the loss from each batch

def plotLoss_new():
    plt.figure(figsize=(10, 8))
    plt.plot(disc_a_history, label='Discriminator A loss')
    plt.plot(disc_b_history, label='Discriminator B loss')
    plt.plot(gen_a2b_history_new, label='Generator a2b loss')
    plt.plot(gen_b2a_history_new, label='Generator b2a loss')
    #plt.plot(cycle_history, label="Cyclic loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/cyclegan_loss.png')
    plt.close()

def saveModels(epoch, genA2B, genB2A, discA, discB):
    genA2B.save('models/generatorA2B_epoch_%d.h5' % epoch)
    genB2A.save('models/generatorB2A_epoch_%d.h5' % epoch)
    discA.save('models/discriminatorA_epoch_%d.h5' % epoch)
    discB.save('models/discriminatorB_epoch_%d.h5' % epoch)


# Training

def train_new(epochs, batch_size, dataset, baselr, use_pseudounet=False, use_unet=False, use_decay=False, plot_models=True):

    lr = baselr
    disc_out_size = 32

    # Load data and normalize
    if dataset=="adiedges4":
        x_train_a, x_train_b, x_test_a, x_test_b, color_a, color_b = loadImagesFromDataset(h, w, dataset, use_hdf5=False)
    else:
        x_train_a, x_train_b, x_test_a, x_test_b = loadImagesFromDataset(h, w, dataset, use_hdf5=False)

    x_train_a = (x_train_a.astype(np.float32) - 127.5) / 127.5
    x_train_b = (x_train_b.astype(np.float32) - 127.5) / 127.5
    x_test_a = (x_test_a.astype(np.float32) - 127.5) / 127.5
    x_test_b = (x_test_b.astype(np.float32) - 127.5) / 127.5

    batchCount_a = x_train_a.shape[0] / batch_size
    batchCount_b = x_train_b.shape[0] / batch_size

    # Train on same image amount, would be best to have even sets
    batchCount = min([batchCount_a, batchCount_b])

    print('\nEpochs:', epochs)
    print('Batch size:', batch_size)
    print('Batches per epoch: ', batchCount, "\n")

    disc_a, disc_b, gen_a2b, gen_b2a = components(w, h, pseudounet=use_pseudounet, unet=use_unet, plot=plot_models)
    saveModels(0, gen_a2b, gen_b2a, disc_a, disc_b)

    pool_a2b = []
    pool_b2a = []

    # Define optimizers
    adam_disc = Adam(lr=baselr, beta_1=0.5)
    adam_gen = Adam(lr=baselr, beta_1=0.5)

    # Define image batches
    true_a = gen_a2b.inputs[0]
    true_b = gen_b2a.inputs[0]

    fake_b = gen_a2b.outputs[0]
    fake_a = gen_b2a.outputs[0]

    fake_pool_a = K.placeholder(shape=(None, 3, h, w))
    fake_pool_b = K.placeholder(shape=(None, 3, h, w))
    # Define labels

    # Labels for generator training
    y_fake_a = K.ones_like(disc_a([fake_a]))
    y_fake_b = K.ones_like(disc_b([fake_b]))

    # Labels for discriminator training
    y_true_a = K.ones_like(disc_a([true_a])) * 0.9
    y_true_b = K.ones_like(disc_b([true_b])) * 0.9

    fakelabel_a2b = K.zeros_like(disc_b([fake_b]))
    fakelabel_b2a = K.zeros_like(disc_a([fake_a]))

    # Labels for color losses
    y_color = K.ones_like(fake_b) 

    # Define losses
    disc_a_loss = mse_loss(y_true_a, disc_a([true_a])) + mse_loss(fakelabel_b2a, disc_a([fake_pool_a]))
    disc_b_loss = mse_loss(y_true_b, disc_b([true_b])) + mse_loss(fakelabel_a2b, disc_b([fake_pool_b]))

    gen_a2b_loss = mse_loss(y_fake_b, disc_b([fake_b]))
    gen_b2a_loss = mse_loss(y_fake_a, disc_a([fake_a]))

    cycle_a_loss = mae_loss(true_a, gen_b2a([fake_b]))
    cycle_b_loss = mae_loss(true_b, gen_a2b([fake_a]))
    cyclic_loss = cycle_a_loss + cycle_b_loss

    # Prepare discriminator updater
    discriminator_weights = disc_a.trainable_weights + disc_b.trainable_weights
    disc_loss = (disc_a_loss + disc_b_loss) * 0.5
    discriminator_updater = adam_disc.get_updates(discriminator_weights, [], disc_loss)

    # Prepare generator updater
    generator_weights = gen_a2b.trainable_weights + gen_b2a.trainable_weights
    gen_loss = (gen_a2b_loss + gen_b2a_loss + lmda * cyclic_loss)
    generator_updater = adam_gen.get_updates(generator_weights, [], gen_loss)

    # Define trainers
    generator_trainer = K.function([true_a, true_b], [gen_a2b_loss, gen_b2a_loss, cyclic_loss], generator_updater)
    discriminator_trainer = K.function([true_a, true_b, fake_pool_a, fake_pool_b], [disc_a_loss/2, disc_b_loss/2], discriminator_updater)

    epoch_counter = 1

    # Start training
    for e in range(1, epochs + 1):
        print('\n','-'*15, 'Epoch %d' % e, '-'*15)

        if use_decay and (epoch_counter > 100):
            lr -= baselr/100
            adam_disc.lr = lr
            adam_gen.lr = lr


        # Initialize progbar and batch counter
        #progbar = generic_utils.Progbar(batchCount)

        np.random.shuffle(x_train_a)
        np.random.shuffle(x_train_b)

        # Cycle through batches
        for i in trange(int(batchCount)):

            # Select true images for training
            #true_batch_a = x_train_a[np.random.randint(0, x_train_a.shape[0], size=batch_size)]
            #true_batch_b = x_train_b[np.random.randint(0, x_train_b.shape[0], size=batch_size)]

            true_batch_a = x_train_a[i*batch_size:i*batch_size+batch_size]
            true_batch_b = x_train_b[i*batch_size:i*batch_size+batch_size]

            # Fake images pool 
            a2b = gen_a2b.predict(true_batch_a)
            b2a = gen_b2a.predict(true_batch_b)

            tmp_b2a = []
            tmp_a2b = []

            for element in a2b:
                if len(pool_a2b) < 50:
                    pool_a2b.append(element)
                    tmp_a2b.append(element)
                else:
                    p = random.uniform(0, 1)

                    if p > 0.5:
                        index = random.randint(0, 49)
                        tmp = np.copy(pool_a2b[index])
                        pool_a2b[index] = element
                        tmp_a2b.append(tmp)
                    else:
                        tmp_a2b.append(element)
            
            for element in b2a:
                if len(pool_b2a) < 50:
                    pool_b2a.append(element)
                    tmp_b2a.append(element)
                else:
                    p = random.uniform(0, 1)

                    if p >0.5:
                        index = random.randint(0, 49)
                        tmp = np.copy(pool_b2a[index])
                        pool_b2a[index] = element
                        tmp_b2a.append(tmp)
                    else:
                        tmp_b2a.append(element)

            pool_a = np.array(tmp_b2a)
            pool_b = np.array(tmp_a2b)

            # Update network and obtain losses
            disc_a_err, disc_b_err = discriminator_trainer([true_batch_a, true_batch_b, pool_a, pool_b])
            gen_a2b_err, gen_b2a_err, cyclic_err = generator_trainer([true_batch_a, true_batch_b])

            # progbar.add(1, values=[
            #                             ("D A", disc_a_err*2),
            #                             ("D B", disc_b_err*2),
            #                             ("G A2B loss", gen_a2b_err),
            #                             ("G B2A loss", gen_b2a_err),
            #                             ("Cyclic loss", cyclic_err)
            #                            ])

        # Save losses for plotting
        disc_a_history.append(disc_a_err)
        disc_b_history.append(disc_b_err)

        gen_a2b_history_new.append(gen_a2b_err)
        gen_b2a_history_new.append(gen_b2a_err)

        #cycle_history.append(cyclic_err[0])
        plotLoss_new()

        plotGeneratedImages(epoch_counter, x_test_a, x_test_b, gen_a2b, gen_b2a)

        if epoch_counter > 150:
            saveModels(epoch_counter, gen_a2b, gen_b2a, disc_a, disc_b)

        epoch_counter += 1


if __name__ == '__main__':
    train_new(200, 1, "horse2zebra", lr, use_decay=True, use_pseudounet=False, use_unet=False, plot_models=False)
