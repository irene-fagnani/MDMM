import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Get the path of the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of 'src' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'src')))

import GMVAE
import networks

def get_z_random(batchSize, nz, num_classes, gpu, random_type='gauss'):
    """
    Sample latent vectors from a mixture of Gaussian distributions.
    :param batchSize: Number of samples.
    :param nz: Dimensionality of the latent space.
    :param random_type: Type of randomness ('gauss' or 'uniform').
    :return: A batch of latent vectors sampled from the mixture.
    """
    num_components = num_classes
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    
    if random_type == 'gauss':
        weights = torch.ones(num_components) / num_components
        categorical = torch.distributions.Categorical(weights)
        component_indices = categorical.sample((batchSize,)).to(device)
        means = torch.randn(num_components, nz).to(device) * 2
        stds = torch.ones(num_components, nz).to(device)
        z = torch.zeros(batchSize, nz).to(device)
        for i in range(num_components):
            mask = (component_indices == i).unsqueeze(1)
            z += mask * torch.normal(means[i], stds[i]).to(device)
        return z

def run_test():
    print("--- Starting consistency test ---")

    # Example Parameters
    batch_size = 2
    input_dim = 3
    z_dim = 108
    y_dim = 3
    c_dim = 2
    img_size = 108
    gaussian_size = 108
    nz = 108
    num_classes = 2
    num_domains = 2
    x_dim = 256
    crop_size = 108
    use_adain = False
    double_layer_ReLUINSConvTranspose = False
    half_size = batch_size // 2

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate your models from the 'networks' module
    content_encoder = networks.MD_E_content(input_dim=input_dim, use_cuda=torch.cuda.is_available()).to(device)
    attr_encoder = networks.MD_E_attr_concat(input_dim=input_dim, z_dim=gaussian_size, y_dim=num_classes, output_nc=nz, c_dim=num_domains, norm_layer=None, nl_layer=networks.get_non_linearity(layer_type='lrelu')).to(device)
    decoder = networks.MD_G_multi_concat(input_dim, x_dim, gaussian_size, crop_size, c_dim=num_domains, nz=nz, use_adain=use_adain, double_ConvT=double_layer_ReLUINSConvTranspose).to(device)

    # Dummy inputs
    x = torch.randn(batch_size, input_dim, img_size, img_size, device=device)
    c = F.one_hot(torch.randint(0, num_domains, (batch_size,)), num_classes=num_domains).float().to(device)

    print("\n--- Model Forward Pass Simulation ---")

    try:
        # Forward pass through content encoder
        content = content_encoder.forward(x)
        z_content_a, z_content_b = torch.split(content, half_size, dim=0)
        print(f"Content A shape: {z_content_a.shape}, Content B shape: {z_content_b.shape}")

        # Forward pass through attribute encoder
        inf = attr_encoder.forward(x, c, temperature=1.0, hard=False)
        z_attr = inf["gaussian"]
        y = inf["categorical"]
        z_attr_a, z_attr_b = torch.split(z_attr, half_size, dim=0)
        z_random = get_z_random(half_size, nz, num_classes, 0)

        # Construct inputs for the generator
        input_content_forA = torch.cat((z_content_b, z_content_a, z_content_b), 0)
        input_content_forB = torch.cat((z_content_a, z_content_b, z_content_a), 0)
        input_attr_forA = torch.cat((z_attr_a, z_attr_a, z_random), 0)
        input_attr_forB = torch.cat((z_attr_b, z_attr_b, z_random), 0)
        y_a = y[0:half_size]
        y_b = y[half_size:]
        input_y_forA = torch.cat((y_a, y_a, y_a), dim=0)
        input_y_forB = torch.cat((y_b, y_b, y_b), dim=0)
        
        # Forward pass through the decoder (generator)
        print("\nAttempting forward pass through the generator...")
        outA = decoder.forward(input_content_forA, input_attr_forA, input_y_forA, y_a)
        outB = decoder.forward(input_content_forB, input_attr_forB, input_y_forB, y_b)

        # --- Consistency Checks ---
        print("\n--- Performing consistency checks ---")
        fake_A_encoded = outA['x_rec']
        fake_B_encoded = outB['x_rec']
        expected_shape = (half_size * 3, input_dim, img_size, img_size)
        assert fake_A_encoded.shape == expected_shape, f"Dimension mismatch for output A. Expected: {expected_shape}, Got: {fake_A_encoded.shape}"
        assert fake_B_encoded.shape == expected_shape, f"Dimension mismatch for output B. Expected: {expected_shape}, Got: {fake_B_encoded.shape}"
        print("Output shapes are correct.")
        assert torch.isfinite(fake_A_encoded).all(), "Output A contains non-finite values (NaN or Inf)."
        assert torch.isfinite(fake_B_encoded).all(), "Output B contains non-finite values (NaN or Inf)."
        print("Outputs do not contain NaN or Inf values.")
        split_A = torch.split(fake_A_encoded, half_size, dim=0)
        split_B = torch.split(fake_B_encoded, half_size, dim=0)
        expected_split_shape = (half_size, input_dim, img_size, img_size)
        assert all(s.shape == expected_split_shape for s in split_A), "Error in output A split."
        assert all(s.shape == expected_split_shape for s in split_B), "Error in output B split."
        print(" Split operation is consistent.")
        print("\n The entire pipeline consistency test passed successfully!")
    
    except Exception as e:
        print(f"\n Test Failed: An unexpected error occurred. {e}")

if __name__ == '__main__':
    run_test()