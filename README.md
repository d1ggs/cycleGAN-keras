# cycleGAN-keras

Fully working keras implementation of cycleGAN in Python, realized during internship at University of Parma's <a  href="http://implab.ce.unipr.it"> IMPLab </a>.

  

\[<a  href="https://github.com/junyanz/CycleGAN"> Authors' repository</a> - <a  href="https://arxiv.org/pdf/1703.10593.pdf">Paper</a>  \]

  

<h2>Samples</h2>

<h3>horse2zebra</h3>

![144-horse2zebra](https://user-images.githubusercontent.com/24715359/48908870-ea066880-ee6b-11e8-905c-47c4278f1c14.png)

<h3>cityscapes</h3>

![198_cityscapes_resnet](https://user-images.githubusercontent.com/24715359/48908826-c511f580-ee6b-11e8-9669-b58685e7171f.png)

<h3>maps</h3>

![195_maps_resnet](https://user-images.githubusercontent.com/24715359/48908829-c6432280-ee6b-11e8-9cf0-cc3286878fa6.png)

<h2>Usage</h2>

To download toy example datasets (check available datasets <a  href="">here</a>) use the datasets script:

`sh datasets.sh dataset_name`  <br>

To enable faster image loading into de network you can use hdf5 format by setting `use_hdf5=True` in `train` function in `cycleGAN.py`. <br>

To build .hdf5 datsets use the make_dataset script in datasets folder:

`python3 make_dataset.py dataset_path number_of_channels`  <br>

For those with limited GPU resources I suggest downscaling the images through the `--img_size` argument (square images are input and output of the network).

  

<h2>Dependencies</h2>

<ul>

<li>Python 3</li>

<li>Tensorflow 1.2 or greater</li>

<li>Keras 2.1.2</li>

<li>keras-contrib</li>

<li>h5py</li>

<li>glob</li>

<li>tqdm</li>

</ul>
