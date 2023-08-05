import numpy as np
from scipy.fftpack import fft, ifft
from scipy.special import hermite, factorial
import matplotlib.pyplot as plt

"""
https://mathoverflow.net/questions/236979/about-eigen-functions-of-the-gaussian-kernel
"""

sigma = 1.0
eps = 8.0
alpha = 1.0

rad = 1 + (2 * eps / alpha)**2


# Define the kernel function
def k(dx):
    return np.exp(-eps**dx**2)

def get_eigenf(n, sigma=sigma):
    Hn = hermite(n, monic=False)
    e = eps
    a = alpha

    psi = (
        lambda x: rad ** (1 / 8)
        / (2**n * factorial(n)) ** 0.5
        * Hn(rad ** (1 / 4)*a*x)
        * np.exp(-(rad**0.5 - 1) * (a**2 * x**2 / 2))
    )
    return psi


def get_lmbd(n):
    e = eps
    a = alpha
    return a * e ** (2 * n) / (((a**2 / 2) * (1 + rad**0.5)) + e**2) ** (n + 1 / 2)


# Invert the eigenvalues (Fourier coefficients)
Nmax = 30
ev = [get_lmbd(n) for n in range(Nmax)]
ev_inv = [1 / lmbd for lmbd in ev]

def k_inv(dx):
    return np.sum([ev_inv_ * get_eigenf(n)(dx) * get_eigenf(n)(0) for n, ev_inv_ in enumerate(ev_inv)], axis=0)

# Define the range for dx
dx = np.linspace(-10, 10, 1000)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot the kernel function
axs[0, 0].plot(dx, k(dx))
axs[0, 0].set_title('Kernel function')

# Plot the eigenvalues
axs[0, 1].plot(range(Nmax), ev, marker='o', linestyle='None')
axs[0, 1].set_title('Eigenvalues')
axs[0, 1].set_yscale('log')

# Plot the inverted eigenvalues
axs[1, 0].plot(range(Nmax), ev_inv, marker='o', linestyle='None')
axs[1, 0].set_title('Inverted eigenvalues')
axs[1, 0].set_yscale('log')

# Plot the inverted kernel function
axs[1, 1].plot(dx, k_inv(dx))
axs[1, 1].set_title('Inverted kernel function')
axs[1, 1].set_yscale('log')


# Set the x-labels for all subplots
for ax in axs.flat:
    ax.set_xlabel('dx')

# Set the y-labels for the left-hand subplots
axs[0, 0].set_ylabel('k')
axs[0, 1].set_ylabel('ev')

# Set the y-labels for the right-hand subplots
axs[1, 0].set_ylabel('ev_inv')
axs[1, 1].set_ylabel('k_inv')

# Add a title to the entire figure
fig.suptitle('Kernel function and eigenvalues')

# Display the plot
plt.show()