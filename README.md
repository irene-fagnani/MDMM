# Harmonization Project.

Pytorch implementation of multi-modality I2I translation for multi-domains. The project is an extension to the "Diverse Image-to-Image Translation via Disentangled Representations(https://arxiv.org/abs/1808.00948)", ECCV 2018, with the integration of GMVAE. 
With the disentangled representation framework, we can learn diverse image-to-image translation among multiple domains.
[[DRIT]](https://github.com/HsinYingLee/DRIT)

Contact: Hsin-Ying Lee (hlee246@ucmerced.edu) and Hung-Yu Tseng (htseng6@ucmerced.edu)

### Prerequisites
- Python 3.5 or Python 3.6
- Pytorch 0.4.0 and torchvision (https://pytorch.org/)
- [TensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- [Tensorflow](https://www.tensorflow.org/) (for tensorboard usage)
- Docker file based on CUDA 9.0, CuDNN 7.1, and Ubuntu 16.04 is provided in the [[DRIT]](https://github.com/HsinYingLee/DRIT) github page.

## Usage
- Training
```
python train.py --dataroot DATAROOT --name NAME --num_domains NUM_DOMAINS --display_dir DISPLAY_DIR --result_dir RESULT_DIR --isDcontent
```

## Datasets
We validate our model on two datasets, available at the following link [datasets](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/)
- apple2orange. Containing two domains: oranges and apples. 
- horse2zebra (mini). Containing two domains: horse and zebra.

The different domains in a dataset should be placed in folders "trainA, trainB, ..." in the alphabetical order.


## Command-line Options

Below is a list of the available command-line arguments added by us and their usage.  

### General
- `--use_cpu`  
  Forces the code to run on the CPU instead of the GPU.  

---

### GMVAE Parameters
- `--num_classes` *(int, default: 2)*  
  Number of classes in the categorical latent space.  

- `--gaussian_size` *(int, default: 108)*  
  Dimensionality of the Gaussian latent space (*z_dim*).  

- `--x_dim` *(int, default: 139968)*  
  Input dimensionality.  
  > Example: `2916 = 3 * 216 * 3 * 3`  

---

### Gumbel Parameters
- `--init_temp` *(float, default: 1.0)*  
  Initial temperature for Gumbel-Softmax (recommended range: 0.5–1.0).  

- `--decay_temp` *(int, default: 1)*  
  Set to `1` to decay the Gumbel temperature every epoch.  

- `--hard_gumbel` *(int, default: 0)*  
  Set to `1` to use the hard version of Gumbel-Softmax.  

- `--min_temp` *(float, default: 0.5)*  
  Minimum temperature after annealing.  

- `--decay_temp_rate` *(float, default: 0.013862944)*  
  Decay rate for the Gumbel temperature per epoch.  

---

### Loss Function Parameters
- `--w_gauss` *(float, default: 1)*  
  Weight of the Gaussian loss term.  

- `--w_categ` *(float, default: 1)*  
  Weight of the categorical loss term.  

- `--w_rec` *(float, default: 1)*  
  Weight of the reconstruction loss term.  

- `--rec_type` *(string, choices: `bce`, `mse`, default: `bce`)*  
  Type of reconstruction loss function to use:  
  - `bce` = Binary Cross-Entropy  
  - `mse` = Mean Squared Error  

---

### Generator / Training Options
- `--use_adain` *(flag)*  
  Enables Adaptive Instance Normalization (AdaIN) in the generator.  

- `--double_layer_ReLUINSConvTranspose` *(flag)*  
  Uses a double ReLU layer in the INSConvTranspose block.  

- `--two_time_scale_update_rule` *(string, default: `none`)*  
  Enables Two Time-Scale Update Rule (TTUR). Options:  
  - `double_gen_enc` = double learning rate for generator & encoder  
  - `half_discr` = half learning rate for discriminator  
  - `none` = disabled  

- `--plot_losses` *(flag)*  
  Plots the loss curves during the first epoch. 

## Note
- The feature transformation (i.e. concat 0) is not fully tested.
- The hyper-parameters matter and are task-dependent. They are not carefully selected yet.
- Our modifications and tests are limited to the training phase.
 

## Papers

["Diverse Image-to-Image Translation via Disentangled Representations"](https://doi.org/10.48550/arXiv.1808.00948)<br> Hsin-Ying Lee,Hung-Yu Tseng, Jia-Bin Huang, Maneesh Kumar Singh, and Ming-Hsuan Yang. <br>
European Conference on Computer Vision (ECCV), 2018 (**oral**).


[“Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders.”](https://doi.org/10.48550/arXiv.1611.02648)<br> Dilokthanakul, Nat, Pedro A. M. Mediano, Marta Garnelo, M. J. Lee, Hugh Salimbeni, Kai Arulkumaran and Murray Shanahan.<br> ArXiv abs/1611.02648 (2016).


