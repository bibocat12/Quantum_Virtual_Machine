# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:25:18 2021

@author: ASUS
"""
from Qsun_jax.Qwave import Wavefunction
import itertools
# allow complex128 / float64 in JAX: enable x64 before importing jax.numpy
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax
from functools import partial

# https://stackoverflow.com/questions/4928297/all-permutations-of-a-binary-sequence-x-bits-long
@partial(jax.jit, static_argnums=(0,))
def Qubit(qubit_num):
    """create a quantum circuit"""
    states = jnp.arange(2**qubit_num)
    amplitude_vector = jnp.zeros(2**qubit_num, dtype=jnp.complex64)
    return Wavefunction(states, amplitude_vector.at[0].set(jnp.complex64(1 + 0j)), qubit_num)

def Walk_Qubit(qubit_num=1, dim=1):
    """create a initial quantum state for hadamard coin"""
    if dim != 1 and dim != 2:
        raise TypeError('The dimension of the quantum walk must be 1 or 2')
    else:
        qubit_num += 1
        if dim == 1:
            #initial state: (|0> - i|1>)x|n=0>/(sqrt(2))
            states = ['0' + str(i) for i in range(2*qubit_num-1)]
            states += ['1' + str(i) for i in range(2*qubit_num-1)]
        
            amplitude_vector = jnp.zeros(4*qubit_num-2, dtype = jnp.complex64)
            amplitude_vector[qubit_num-1] = 2**-0.5
            amplitude_vector[3*qubit_num-2] = (-2)**-0.5
            return Wavefunction(states, amplitude_vector)
        else:
            #initial state: ((|0> + i|1>)/sqrt(2))x((|0> + i|1>)/sqrt(2))x|n=0>x|n=0>
            states = ['0' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            states += ['1' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            states += ['2' + str(i) for i in range(0, (2*qubit_num-1)**2)]
            states += ['3' + str(i) for i in range(0, (2*qubit_num-1)**2)]

            amplitude_vector = jnp.zeros(4*(2*qubit_num-1)**2, dtype = jnp.complex64)
            index = int(((2*qubit_num-1)**2-1)/2)
            amplitude_vector[index] = 1/2
            amplitude_vector[index+(2*qubit_num-1)**2] = 0.5j
            amplitude_vector[index+2*(2*qubit_num-1)**2] = 0.5j
            amplitude_vector[index+3*(2*qubit_num-1)**2] = -1/2
            return Wavefunction(states, amplitude_vector)