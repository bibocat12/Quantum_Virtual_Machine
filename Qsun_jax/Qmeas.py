# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 17:08:42 2021

@author: ASUS
"""
import jax.numpy as jnp
import jax
import numpy
import math



def measure_all(wavefunction, n_samples):
    """Make a measurement on quibits"""
#    inds_onp = numpy.random.choice(len(wavefunction.state), n_samples, p=wavefunction.probabilities())
    probs = jnp.array(wavefunction.probabilities())
    

    key = jax.random.PRNGKey(40)
    inds_jax = jax.random.choice(key, a=len(wavefunction.state), shape=(n_samples,), p=probs)
    
    # Convert JAX indices to numpy integers to index the Python list
    inds_host = numpy.array(inds_jax, dtype=int)
    state_arr = numpy.array(wavefunction.state)
    return numpy.unique(state_arr[inds_host], return_counts=True)

@jax.jit
def measure_one(wavefunction, n):
    """return a probability of |0> and |1> of qubit n"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = wavefunction.n_qubits
    # NOTE: Avoid Python boolean checks on traced values (n may be a JAX tracer)
    # Caller ensures 0 <= n < qubit_num. Compute target bits vectorized over states.
    # states is an integer JAX array of shape (2**qubit_num,)
    target_bits = (states >> (qubit_num - n - 1)) & 1
    
    # Calculate probability using vectorized operations
    amplitude_squared = jnp.abs(amplitude) ** 2
    
    # Sum probabilities where target bit is 0
    prob_0 = jnp.sum(jnp.where(target_bits == 0, amplitude_squared, 0.0))
    prob_0 = jnp.round(prob_0, 10)
    
    return jnp.array([prob_0, 1.0 - prob_0])

def collapse_one(wavefunction, n):
    """Measurement operator which make the Nth qubit collapse into |0> or |1>"""
    """regular error: https://stackoverflow.com/questions/48017053/numpy-random-choice-function-gives-weird-results"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    prob_0 = 0
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in range(2**qubit_num):
        if states[i][n] == '0':
            prob_0 += abs(amplitude[i])**2
    result_measure = jnp.random.choice(['0', '1'], 1, p = [prob_0, 1 - prob_0])[0]
    if result_measure == '0':
        for i in range(2**qubit_num):
            if states[i][n] == '0':
                new_amplitude[i] = amplitude[i]/math.sqrt(prob_0)
    elif result_measure == '1':
        for i in range(2**qubit_num):
            if states[i][n] == '1':
                new_amplitude[i] = amplitude[i]/math.sqrt(1 - prob_0)
    wavefunction.amplitude = new_amplitude