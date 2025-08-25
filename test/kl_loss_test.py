import sys
import os
# Get the path of the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory of 'src' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..', 'src')))

import torch
import torch.nn.functional as F
import GMVAE




def test_gaussian_loss():
    """
    Tests the gaussian_loss function from the GMVAE module.

    The test creates two identical Gaussian distributions and calculates the
    Kullback-Leibler (KL) divergence between them. The expected result is a
    value close to zero, which is verified using a torch.isclose assertion.
    """
    print("Executing the gaussian_loss function test...")

    # KL(N(0,1) || N(0,1)) should be close to 0.
    # Create identical distributions.
    mu = torch.tensor([0.0])
    var = torch.tensor([1.0])
    mu_prior = torch.tensor([0.0])
    var_prior = torch.tensor([1.0])

    z = mu + torch.sqrt(var) * torch.randn_like(mu)

    # Instantiate the loss calculator
    loss_calculator = GMVAE.LossFunctions()
    kl_loss_calculated = loss_calculator.gaussian_loss(z, mu, var, mu_prior, var_prior)

    expected_kl_approx = 0.0

    assert torch.isclose(kl_loss_calculated, torch.tensor(expected_kl_approx), rtol=1e-2), \
        f"Test Failed! Expected value: {expected_kl_approx}, Obtained value: {kl_loss_calculated.item()}"

    print(f"Test Passed. Calculated KL value: {kl_loss_calculated.item()}")
    print("KL divergence between identical distributions is correctly close to 0.")


if __name__ == '__main__':
    test_gaussian_loss()