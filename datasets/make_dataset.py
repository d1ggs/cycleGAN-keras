import os
import cv2
import h5py
import parmap
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pylab as plt


def format_image(img_path, size, nb_channels):
    """
    Load img with opencv and reshape
    """

    if nb_channels == 1:
        img = cv2.imread(img_path, 0)
        img = np.expand_dims(img, axis=-1)
    else:
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # GBR to RGB

    w = img.shape[1]

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if nb_channels == 1:
        img = np.expand_dims(img, -1)

    img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)

    return img


def build_HDF5(jpeg_dir, nb_channels, set_lenght = 0,size=256):
    """
    Gather the data in a single HDF5 file.
    """

    # Put train data in HDF5
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)
    with h5py.File(hdf5_file, "w") as hfw:

        for dset_type in ["trainA", "testA", "trainB", "testB"]:
            #import .jpg files
            list_img = [img for img in Path(jpeg_dir).glob('%s/*.jpg' % dset_type)]
            list_img = [str(img) for img in list_img]
            
            #import .png files
            list_img.extend(list(Path(jpeg_dir).glob('%s/*.png' % dset_type)))
            list_img = list(map(str, list_img))
            list_img = np.array(list_img)
            
            ## Shrink dataset size if specified with command line argument
            if ("train" in dset_type) and (set_lenght != 0) and (set_lenght < len(list_img)):
                list_img = list_img[:set_lenght]

            data = hfw.create_dataset("%s_data" % dset_type,
                                           (0, nb_channels, size, size),
                                           maxshape=(None, 3, size, size),
                                           dtype=np.uint8)

            num_files = len(list_img)
            chunk_size = 100
            num_chunks = num_files / chunk_size
            if num_chunks < 1:
                num_chunks = 1
            arr_chunks = np.array_split(np.arange(num_files), num_chunks)

            for chunk_idx in tqdm(arr_chunks):

                list_img_path = list_img[chunk_idx].tolist()
                output = parmap.map(format_image, list_img_path, size, nb_channels, pm_parallel=False)

                arr_img = np.concatenate([o for o in output], axis=0)

                # Resize HDF5 dataset
                data.resize(data.shape[0] + arr_img.shape[0], axis=0)

                data[-arr_img.shape[0]:] = arr_img.astype(np.uint8)


def check_HDF5(jpeg_dir, nb_channels):
    """
    Plot images with landmarks to check the processing
    """

    # Get hdf5 file
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)

    with h5py.File(hdf5_file, "r") as hf:
        data = hf["trainA_data"]
        for i in range(5):
            plt.figure()
            print(data.shape)
            img = data[i, :, :, :].transpose(1,2,0)
            print(img.shape)
            if nb_channels == 1:
                plt.imshow(img[:, :, 0], cmap="gray")
            else:
                plt.imshow(img)
            plt.show()
            plt.clf()
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build dataset')
    parser.add_argument('jpeg_dir', type=str, help='path to jpeg images')
    parser.add_argument('nb_channels', type=int, help='number of image channels')
    parser.add_argument('--set_size', type=int, help='number of images per class')
    parser.add_argument('--img_size', default=256, type=int,
                        help='Desired Width == Height')
    parser.add_argument('--do_plot', action="store_true",
                        help='Plot the images to make sure the data processing went OK')
    args = parser.parse_args()

    data_dir = "./processed"

    build_HDF5(args.jpeg_dir, args.nb_channels, size=args.img_size, set_lenght=args.set_size)

    if args.do_plot:
        check_HDF5(args.jpeg_dir, args.nb_channels)
