# Deconstructing Images: A Generative Approach.

Pytorch implementation of multi-modality I2I translation for multi-domains. The project is an extension to the "Diverse Image-to-Image Translation via Disentangled Representations(https://arxiv.org/abs/1808.00948)", ECCV 2018, with the integration of a Gaussian Mixture Variational Autoencoder (https://doi.org/10.48550/arXiv.1611.02648). 
With the disentangled representation framework, we can learn diverse image-to-image translation among multiple domains.
[[DRIT]](https://github.com/HsinYingLee/DRIT)


This project explores integrating a *Gaussian Mixture Variational Autoencoder (GMVAE)* into an image-to-image (I2I) translation framework. Our goal is to overcome the limitations of standard VAEs that assume a uni-modal latent space, which restricts a model's ability to capture the full diversity of target domain images.

### The Problem

Existing I2I translation models, such as DRIT++, often rely on standard VAEs that use a single Gaussian distribution in their latent space. This assumption is a limitation when data is inherently multi-modal, as it restricts the model's ability to represent complex variations. This can lead to a lack of diversity and expressiveness in translated images.

### Our Solution

We propose a novel approach that replaces the standard VAE with a GMVAE. By modeling the latent space with a *mixture of Gaussian distributions*, our framework provides a more flexible and structured representation. This allows the model to better capture and learn the rich, multi-modal variations within the target domain.

### Project Objectives

The primary objective of this work is to investigate whether using a GMVAE can lead to measurable improvements in I2I translation tasks. Specifically, we evaluate how this modification influences the quality and variability of translated images, and whether it effectively addresses the limitations of prior work.


### Prerequisites

The prerequisites are stored in the file "requirements.txt", under the folder "docs". To install them: 

```
pip install -r docs/requirements.txt

```


## Usage
- Training
```
python src/train.py --dataroot DATAROOT --name NAME --num_domains NUM_DOMAINS --display_dir DISPLAY_DIR --result_dir RESULT_DIR --isDcontent
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

## Training Script: `run_train.sh`

A convenient shell script is provided to run training with dataset-specific defaults and optional flags for experimental control.

### Usage

```bash
./run_train.sh DATASET_NAME [MAX_IT] [--use_cpu] [--use_adain] [--double_layer] [--two_time_scale_rule RULE]
```

### Examples

```bash
# Train summer2winter_yosemite with 5000 iterations on GPU
./run_train.sh summer2winter_yosemite 5000

# Train summer2winter_yosemite with 5000 iterations on CPU
./run_train.sh summer2winter_yosemite 5000 --use_cpu

# Train mini with 10000 iterations, AdaIN, double conv layers, and two-time scale update
./run_train.sh mini 10000 --use_adain --double_layer double_gen_enc

# Train orange2apple with default settings
./run_train.sh orange2apple
```

### Experiments related to our work:

#### Loss integration (ch. 6.2)
```bash
./run_train.sh apple2orange 5000
```

#### Addressing the Discriminator-Generator Imbalance, AdaIN (ch. 6.3)

```bash
./run_train.sh apple2orange 4500 --use_adain
```

#### Addressing the Discriminator-Generator Imbalance, Transposed Convolutional Layer (ch. 6.3)

```bash
./run_train.sh apple2orange 4000 --double_layer
```

#### Addressing the Discriminator-Generator Imbalance, Halving the Discriminator’s Learning Rate (ch. 6.3)

```bash
./run_train.sh apple2orange 5500 --two_time_scale_update_rule half_discr
```

#### Addressing the Discriminator-Generator Imbalance, Doubling the Generator’s Learning Rate (ch. 6.3)

```bash
./run_train.sh apple2orange 5500 --two_time_scale_update_rule double_gen_enc
```



Before running, make sure the script is executable:

```bash
chmod +x run_train.sh
```

## Test Folder

The `test` folder contains three scripts designed to validate the implementation and functionality of our contributions within the existing framework.

Specifically:

- **`kl_loss_test`**: evaluates the effectiveness of the Gaussian loss in modeling multimodal distributions.  
- **`consistency_test`**: ensures that the encoder and decoder operate coherently.  
- **`integration_test`**: runs the full pipeline on a smaller scale (fewer epochs and iterations) using a toy dataset, allowing us to observe how different components interact.

To run all three tests at once, use the provided `run_tests.sh` script. First, ensure it is executable:

```bash
chmod +x run_tests.sh
```


## Note
- The feature transformation (i.e. concat 0) is not fully tested.
- The hyper-parameters matter and are task-dependent. They are not carefully selected yet.
- Our modifications and tests are limited to the training phase.
 

## Papers

["Diverse Image-to-Image Translation via Disentangled Representations"](https://doi.org/10.48550/arXiv.1808.00948)<br> Hsin-Ying Lee,Hung-Yu Tseng, Jia-Bin Huang, Maneesh Kumar Singh, and Ming-Hsuan Yang. <br>
European Conference on Computer Vision (ECCV), 2018 (**oral**).


[“Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders.”](https://doi.org/10.48550/arXiv.1611.02648)<br> Dilokthanakul, Nat, Pedro A. M. Mediano, Marta Garnelo, M. J. Lee, Hugh Salimbeni, Kai Arulkumaran and Murray Shanahan.<br> ArXiv abs/1611.02648 (2016).

"Deconstructing Images: A Generative Approach."<br> Irene Fagnani, Greta Gorbani (2025).


