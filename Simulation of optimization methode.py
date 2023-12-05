import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.optimize import minimize
from scipy.signal import convolve, square, sawtooth
import matplotlib.pyplot as plt


# Simulate a signal with noise
def simulate_signal(t, params):
    square_wave1 = square(2 * jnp.pi * 5 * t)
    sinusoidal_wave1 = jnp.sin(2 * jnp.pi * 3 * t)
    signal1 = convolve(square_wave1, sinusoidal_wave1, mode='same', method='auto')

    square_wave2 = sawtooth(2 * jnp.pi * 2 * t)
    sinusoidal_wave2 = jnp.sin(2 * jnp.pi * 4 * t)
    signal2 = convolve(square_wave2, sinusoidal_wave2, mode='same', method='auto')

    # Generate random noise using numpy instead of jax.numpy
    noise = 0.1 * np.random.normal(size=len(t))

    noisy_signal = signal1 + signal2 + noise
    return noisy_signal


# Theoretical model
def theoretical_model(t):
    square_wave1 = square(2 * jnp.pi * 5 * t)
    sinusoidal_wave1 = jnp.sin(2 * jnp.pi * 3 * t)
    signal1 = convolve(square_wave1, sinusoidal_wave1, mode='same', method='auto')

    square_wave2 = sawtooth(2 * jnp.pi * 2 * t)
    sinusoidal_wave2 = jnp.sin(2 * jnp.pi * 4 * t)
    signal2 = convolve(square_wave2, sinusoidal_wave2, mode='same', method='auto')

    theoretical_signal = signal1 + signal2
    return theoretical_signal

# Absolute error function
def absolute_error(params, t, noisy_signal):
    square_wave1, sinusoidal_wave1, square_wave2, sinusoidal_wave2, noise_level = params
    model_signal = convolve(square_wave1, sinusoidal_wave1, mode='same', method='auto') + \
                   convolve(square_wave2, sinusoidal_wave2, mode='same', method='auto') + \
                   noise_level * jnp.random.normal(size=len(t))
    error = jnp.sum(jnp.abs(model_signal - noisy_signal))
    return error

# Function to calculate absolute error
def jax_absolute_error(params, t, noisy_signal):
    signal = simulate_signal(t, params)
    return jnp.sum(jnp.abs(signal - noisy_signal))

# Function to optimize parameters
def optimize_params(absolute_error, initial_params, t, noisy_signal):
    result = minimize(absolute_error, initial_params, args=(t, noisy_signal), method='Nelder-Mead')
    optimized_params = result.x
    return optimized_params

# Generate synthetic data
t = jnp.linspace(0, 10, 1000)
params1 = (1.0, 2.0)  # Example parameters for signal 1
params2 = (0.5, 1.5)  # Example parameters for signal 2
noise_level = 0.1
noisy_signal = simulate_signal(t, (params1, params2, noise_level)) + noise_level * jax.random.normal(jax.random.PRNGKey(0), shape=(len(t),))

# Initial parameters for optimization
initial_params = (params1, params2, noise_level)

# Optimize parameters
optimized_params = optimize_params(jax_absolute_error, initial_params, t, noisy_signal)

# Print the optimized parameters
print("Optimized Parameters:", optimized_params)


# Plot the results
plt.figure(figsize=(12, 6))

# Plot the noisy signal
plt.subplot(2, 1, 1)
plt.plot(t, noisy_signal, label="Noisy Signal")
plt.title("Noisy Signal")




# Plot the optimized signal
optimized_signal = jnp.convolve(jnp.sign(jnp.sin(2 * jnp.pi * optimized_params[0] * t)),
                                jnp.sin(2 * jnp.pi * optimized_params[1] * t), mode='same') + \
                  jnp.convolve(jnp.sign(jnp.sin(2 * jnp.pi * optimized_params[2] * t)),
                                jnp.sin(2 * jnp.pi * optimized_params[3] * t), mode='same')
plt.subplot(2, 1, 2)
plt.plot(t, optimized_signal, label="Optimized Signal", color='orange')
plt.title("Optimized Signal")

plt.tight_layout()
plt.show()


# # Function to calculate absolute error
# def jax_absolute_error(params, t, noisy_signal):
#     signal = simulate_signal(t, params)
#     return jnp.sum(jnp.abs(signal - noisy_signal))

# # Function to optimize parameters
# def optimize_params(absolute_error, initial_params, t, noisy_signal):
#     result = minimize(absolute_error, initial_params, args=(t, noisy_signal), method='BFGS')
#     optimized_params = result.x
#     return optimized_params

# # Generate synthetic data
# t = jnp.linspace(0, 10, 1000)
# params1 = (1.0, 2.0)  # Example parameters for signal 1
# params2 = (0.5, 1.5)  # Example parameters for signal 2
# noise_level = 0.1
# noisy_signal = simulate_signal(t, (params1, params2, noise_level)) + noise_level * jax.random.normal(jax.random.PRNGKey(0), shape=(len(t),))

# # Initial parameters for optimization
# initial_params = (params1, params2, noise_level)

# # Optimize parameters using vmap
# vmap_absolute_error = vmap(jax_absolute_error, in_axes=(None, 0, 0))
# optimized_params = minimize(lambda params: vmap_absolute_error(params, t, noisy_signal), initial_params, method='BFGS').x

# # Print the optimized parameters
# print("Optimized Parameters:", optimized_params)

# # Plot the results
# plt.plot(t, noisy_signal, label='Noisy Signal')
# plt.plot(t, simulate_signal(t, optimized_params), label='Optimized Signal')
# plt.legend()
# plt.show()