# VQVAE HipMRI Pattern Recognition + Recognition


## Description
This repository contains an implementation of the Vector Quantized Variational AutoEncoder (VQVAE) for generative modelling and reconstruction of HipMRI slices from the CSIRO HipMRI study. The goal is to learn a discrete latent representation of MRI images that enables:
- High-fidelity reconstruction of MRI slices
- Generation of new plausible MRI samples
- Quantitative assesement using SSIM (Structural Similarity Index). To this, an acceptable result will be over 0.6.
The original paper for VQVAEs can be found at https://arxiv.org/abs/1711.00937.

## VQVAE Description
A VQVAE is a generative machine learning model that aims to improve on variational autoencoders by adding a Vector Quantization layer. A VAE learns to compress images into a continuous latent space where we can generate new data from. Contrarily, VQVAE's learns to compress images into a discrete latent space where each part of an image is represented by a codebook of vectors (this process is called vector quantization). Generally, this makes reconstructions sharper and more detailed than VAE's, because it chooses from specific learned patterns rather than blending everything in an average.


## Implementation
Generally, the architecture of a VQVAE is:

![VQVAE Architecture](readme_images/image-2.png)

Here, we see a conventional implementation. There are 3 components to take note of: an Encoder, Vector Quantizer (middle) and Decoder. These are explained in detail below.

#### Encoder + Decoder
The Encoder takes an input image x and compresses it into a smaller, latent space through downsampling by convolutional layers and batch normalization. Essentially maps an input $x -> z$ latent space. After passing image data through many downsampling layers, residual blocks recieve feature maps for refinement. 

The Decoder does the opposite, it takes the quantized latent codes from the vector quantizer and reconstructs the MRI image, allowing the model to learn how to generate realistic hip MRI patterns (takes $z -> x$)

#### Vector Quantizer
Turns encoder's continuous features into discrete codes chosen from a learning codebook (set of embedding vectors). This is a bottleneck with discrete symbols, helping the model learn a compact, reusable vocabulary of patterns (i.e. MRI textures/shapes). Essentially, it:
    - Computes distances from each latent vector to all embedding vectors and picks the nearest using a one hot index.
    - Loss is the codebook loss and it moves codes toward the encoder outputs:
        $β ||z_e – sg(z_q)||² $ from a hyperparameter β. 

### Residual Layer + Stack
A ResidualLayer is single residual block that refines features without losing the original signal. It learns a small correction and adds it back to the input (x + f(x)), which stabilizes training and helps go deeper. It utilises a ReLU activation layer with a 3x3 and 1x1 convolution layer which helps the model learn detail refinements (i.e. edges and small structures). 

A ResidualStack stacks these multiple ResidualLayer blocks in sequence to increase capacity, which lets the encoder/decoder model increasingly complex patterns without exploding or vanishing gradients. 

#### Contributions
The idea for the VQVAE in project came from MishaLaskin, whose original code can be found here: 
https://github.com/MishaLaskin/vqvae
Additionally, the original paper for VQVAEs can be found at https://arxiv.org/abs/1711.00937.

## Training

## Image Generation

## Reproduction

### Dependencies

### Justify Hyperparams

## Architecture
