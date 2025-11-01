    # -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:26:23 2021

@author: ASUS
"""
import jax
import jax.numpy as jnp
import cmath
from jax import lax
from Qsun_jax.Qwave import _wf_flatten

@jax.jit
def H(wavefunction, n):
    """Hadamard gate"""
    children, aux = wavefunction.tree_flatten()
    states = children[0]
    amplitude = children[1]
    qubit_num = aux
    new_amplitude = jnp.zeros(2**qubit_num, dtype = jnp.complex64)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
   # print(f"Amplitudes changing on qubit {n}: {amplitude}, {jnp.nonzero(amplitude)[0]}")
    for i in jnp.nonzero(amplitude)[0]:
    #    print(states[i][n])
        if states[i][n] == '0':
            cur_amplitude = amplitude[i]/2**0.5
            
            new_amplitude = new_amplitude.at[i].set(cur_amplitude)
            new_amplitude = new_amplitude.at[i+cut].set(new_amplitude[i+cut] + cur_amplitude)
        else:
            new_amplitude = new_amplitude.at[i].set(new_amplitude[i] - amplitude[i]/2**0.5)
            new_amplitude = new_amplitude.at[i-cut].set(new_amplitude[i-cut] + amplitude[i]/2**0.5)
   # print(new_amplitude)
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'H'])
    
def X(wavefunction, n):
    """Pauli-X"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in jnp.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i+cut] += amplitude[i]
        else:
            new_amplitude[i-cut] += amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'X'])
    
def Y(wavefunction, n):
    """Pauli-Y"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in jnp.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i+cut] += 1.0j*amplitude[i]
        else:
            new_amplitude[i-cut] -= 1.0j*amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'Y'])
    
def Z(wavefunction, n):
    """Pauli-Z"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in jnp.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += amplitude[i]
        else:
            new_amplitude[i] -= amplitude[i]  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'Z'])
    

@jax.jit
def RX(wavefunction, n, phi=0):
    """Rotation around X-axis gate"""
    children, aux = wavefunction.tree_flatten()
    states = children[0]
    amplitude = children[1]
    qubit_num = aux
    new_amplitude = jnp.zeros(2**qubit_num, dtype = jnp.complex64)
    cut = 2**(qubit_num-n-1)
    
    cos_term = jnp.cos(phi / 2)
    sin_term = jnp.sin(phi / 2)
    state_bits = jnp.array([(s >> (qubit_num - n - 1)) & 1 for s in states])
    
    def body_fun(i, new_amp):
        bit = state_bits[i]
        #jax.debug.print("Processing state index {} with bit {}", i, bit)
        def apply_zero(new_amp):
            idx0 = i
            idx1 = i + cut
            new_amp = new_amp.at[idx0].set(new_amp[idx0] + cos_term * amplitude[i])
            new_amp = new_amp.at[idx1].set(new_amp[idx1] + -1j * sin_term * amplitude[i])
            #jax.debug.print("Applied RX on index {}: idx0 {}, idx1 {}", i, new_amp,  amplitude[i])
            return new_amp

        def apply_one(new_amp):
            idx0 = i
            idx1 = i - cut
            new_amp = new_amp.at[idx0].set(new_amp[idx0] + cos_term * amplitude[i])
            new_amp = new_amp.at[idx1].set(new_amp[idx1] + -1j * sin_term * amplitude[i])
            #jax.debug.print("Applied RX on index {}: idx0 {}, idx1 {}", i, new_amp,  amplitude[i])
            return new_amp

        return lax.cond(bit == 0, apply_zero, apply_one, new_amp)

    new_amplitude = lax.fori_loop(0, 2**qubit_num, body_fun, jnp.zeros_like(amplitude))
    wavefunction.amplitude = new_amplitude
#    wavefunction = wavefunction.replace(amplitude=new_amplitude)
    # jax.debug.print("RX gate on qubit {}", wavefunction.amplitude)
    #wavefunction.visual.append([n, 'RX', '0'])
    return wavefunction
    

@jax.jit
def RY(wavefunction, n, phi=0):
    """Rotation around Y-axis gate"""
    children, aux = wavefunction.tree_flatten()
    states = children[0]
    amplitude = children[1]
    qubit_num = aux
    new_amplitude = jnp.zeros(2**qubit_num, dtype = jnp.complex64)
    cut = 2**(qubit_num-n-1)
    # if n >= qubit_num or n < 0:
    #     raise TypeError("Index is out of range")

    cos_term = jnp.cos(phi / 2)
    sin_term = jnp.sin(phi / 2)
    state_bits = jnp.array([ (s >> (qubit_num - n)) & 1 for s in states]) 
    def body_fun(i, new_amp):
        bit = state_bits[i]
        
        def apply_zero(new_amp):
            idx0 = i
            idx1 = i + cut
            new_amp = new_amp.at[idx0].set(new_amp[idx0] + cos_term * amplitude[i])
            new_amp = new_amp.at[idx1].set(new_amp[idx1] + sin_term * amplitude[i])
            return new_amp

        def apply_one(new_amp):
            idx0 = i
            idx1 = i - cut
            new_amp = new_amp.at[idx0].set(new_amp[idx0] + cos_term * amplitude[i])
            new_amp = new_amp.at[idx1].set(new_amp[idx1] + -sin_term * amplitude[i])
            return new_amp

        return lax.cond(bit == 0, apply_zero, apply_one, new_amp)

    new_amplitude = lax.fori_loop(0, 2**qubit_num, body_fun, jnp.zeros_like(amplitude))
    wavefunction.amplitude = new_amplitude
    # jax.debug.print("RY gate on qubit {}", n)
    #wavefunction.visual.append([n, 'RY', '0'])
    return wavefunction


@jax.jit
def RZ(wavefunction, n, phi=0):
    """Rotation around Z-axis gate"""
    children, aux = wavefunction.tree_flatten()
    states = children[0]
    amplitude = children[1]
    qubit_num = aux
    new_amplitude = jnp.zeros(2**qubit_num, dtype = jnp.complex64)
    
    exp_neg = jnp.exp(-1j * phi / 2)
    exp_pos = jnp.exp(1j * phi / 2)
    state_bits = jnp.array([(s >> (qubit_num - n - 1)) & 1 for s in states])
    
    def body_fun(i, new_amp):
        bit = state_bits[i]
        
        def apply_zero(new_amp):
            new_amp = new_amp.at[i].set(new_amp[i] + exp_neg * amplitude[i])
            return new_amp

        def apply_one(new_amp):
            new_amp = new_amp.at[i].set(new_amp[i] + exp_pos * amplitude[i])
            return new_amp

        return lax.cond(bit == 0, apply_zero, apply_one, new_amp)

    new_amplitude = lax.fori_loop(0, 2**qubit_num, body_fun, jnp.zeros_like(amplitude))
    wavefunction.amplitude = new_amplitude
#    wavefunction = wavefunction.replace(amplitude=new_amplitude)
    #   
    #wavefunction.visual.append([n, 'RZ', '0'])
    return wavefunction
    
def Phase(wavefunction, n, phi=0):
    """PHASE gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in jnp.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i] += amplitude[i]
        else:
            new_amplitude[i] += cmath.exp(1j*phi)*amplitude[i]  
    wavefunction.amplitude = new_amplitude
#     (wavefunction.visual).append([n, 'P', phi])
    
def S(wavefunction, n):
    """Phase(pi/2)"""
    Phase(wavefunction, n , cmath.pi/2)
    (wavefunction.visual).append([n, 'S'])
    
def T(wavefunction, n):
    """Phase(pi/4)"""
    Phase(wavefunction, n , cmath.pi/4)
    (wavefunction.visual).append([n, 'T'])
    
def Xsquare(wavefunction, n):
    """a square root of the NOT gate."""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in jnp.nonzero(amplitude)[0]:
        new_amplitude[i] += (1+1j)*amplitude[i]/2
        if states[i][n] == '0':
            new_amplitude[i+cut] += (1-1j)*amplitude[i]/2
        else:
            new_amplitude[i-cut] += (1-1j)*amplitude[i]/2  
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([n, 'XS'])


@jax.jit
def CNOT(wavefunction, control, target):
    """Flip target if control is |1>"""
    children, aux = wavefunction.tree_flatten()
    states = children[0]
    amplitude = children[1]
    qubit_num = aux
    new_amplitude = jnp.zeros(2**qubit_num, dtype = jnp.complex64)
    
    # if control == target:
    #     raise TypeError("Control qubit and target qubit must be distinct")
    
    cut = 2**(qubit_num-target-1)
    
    # Pre-compute control and target bits for all states
    control_bits = jnp.array([(s >> (qubit_num - control - 1)) & 1 for s in states])
    target_bits = jnp.array([(s >> (qubit_num - target - 1)) & 1 for s in states])
    
    def body_fun(i, new_amp):
        control_bit = control_bits[i]
        target_bit = target_bits[i]
        
        def control_is_one(new_amp):
            # If control is |1>, flip target
            def target_is_zero(new_amp):
                # Flip from |0> to |1>
                new_amp = new_amp.at[i + cut].set(new_amp[i + cut] + amplitude[i])
                return new_amp
            
            def target_is_one(new_amp):
                # Flip from |1> to |0>
                new_amp = new_amp.at[i - cut].set(new_amp[i - cut] + amplitude[i])
                return new_amp
            
            return lax.cond(target_bit == 0, target_is_zero, target_is_one, new_amp)
        
        def control_is_zero(new_amp):
            # If control is |0>, no change
            new_amp = new_amp.at[i].set(amplitude[i])
            return new_amp
        
        return lax.cond(control_bit == 1, control_is_one, control_is_zero, new_amp)
    
    new_amplitude = lax.fori_loop(0, 2**qubit_num, body_fun, jnp.zeros_like(amplitude))
    wavefunction.amplitude = new_amplitude
    # wavefunction = wavefunction.replace(amplitude=new_amplitude)
    #wavefunction.visual.append([control, target, 'CX'])
    return wavefunction
    
def CPhase(wavefunction, control, target, phi=0):
    """Controlled PHASE gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    if control == target:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in jnp.nonzero(amplitude)[0]:
        if states[i][control] == '1':
            if states[i][target] == '0':
                new_amplitude[i] += amplitude[i]
            else:
                new_amplitude[i] += cmath.exp(1j*phi)*amplitude[i] 
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control, target, 'CP', '0'])
    
def CCNOT(wavefunction, control_1, control_2, target):
    """CCNOT - double-controlled-X"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-target-1)
    if control_1 == target or control_2 == target or control_1 == control_2:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in jnp.nonzero(amplitude)[0]:
        if states[i][control_1] == '1' and states[i][control_2] == '1':
            if states[i][target] == '0':
                new_amplitude[i+cut] += amplitude[i]
            else:
                new_amplitude[i-cut] += amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([control_1, control_2, target, 'CCX'])
    
def OR(wavefunction, control_1, control_2, target):
    """CCNOT - double-controlled-X"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-target-1)
    if control_1 == target or control_2 == target or control_1 == control_2:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in jnp.nonzero(amplitude)[0]:
        if states[i][control_1] == '1' or states[i][control_2] == '1':
            if states[i][target] == '0':
                new_amplitude[i+cut] += amplitude[i]
            else:
                new_amplitude[i-cut] += amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    
def SWAP(wavefunction, target_1, target_2):
    """Swap gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    minimum = target_2 ^ ((target_1 ^ target_2) & -(target_1 < target_2))
    maximum = target_1 ^ ((target_1 ^ target_2) & -(target_1 < target_2)) 
    cut = 2**(qubit_num-minimum-1) - 2**(qubit_num-maximum-1)
    if target_1 == target_2:
        raise TypeError("Target qubits must be distinct")
    for i in range(2**qubit_num):
        if states[i][target_1] != states[i][target_2]:
            if int(states[i][maximum]) > int(states[i][minimum]):
#                 print(states[i], 'to', states[i+cut])
                new_amplitude[i+cut] += amplitude[i]                              
            else:
#                 print(states[i], 'to', states[i-cut])
                new_amplitude[i-cut] += amplitude[i]
        else:
#                 print(states[i], 'to', states[i])
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([target_1, target_2, 'SWAP'])

def CSWAP(wavefunction, control, target_1, target_2):
    """CSwap gate"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    minimum = target_2 ^ ((target_1 ^ target_2) & -(target_1 < target_2))
    maximum = target_1 ^ ((target_1 ^ target_2) & -(target_1 < target_2)) 
    cut = 2**(qubit_num-minimum-1) - 2**(qubit_num-maximum-1)
    if control == target_1 or control == target_2 or target_1 == target_2:
        raise TypeError("Control qubit and target qubit must be distinct")
    for i in range(2**qubit_num):
        if states[i][control] == '1':
            if states[i][target_1] != states[i][target_2]:
                if int(states[i][maximum]) > int(states[i][minimum]):
    #                 print(states[i], 'to', states[i+cut])
                    new_amplitude[i+cut] += amplitude[i]                              
                else:
    #                 print(states[i], 'to', states[i-cut])
                    new_amplitude[i-cut] += amplitude[i]
            else:
    #                 print(states[i], 'to', states[i])
                new_amplitude[i] = amplitude[i]
        else:
            new_amplitude[i] = amplitude[i]
    wavefunction.amplitude = new_amplitude
    (wavefunction.visual).append([target_1, target_2, control, 'CSWAP'])
    
def E(wavefunction, p, n):
    """Quantum depolarizing channel"""
    states = wavefunction.state
    amplitude = wavefunction.amplitude
    qubit_num = len(states[0])
    new_amplitude = jnp.zeros(2**qubit_num, dtype = complex)
    cut = 2**(qubit_num-n-1)
    if n >= qubit_num or n < 0:
        raise TypeError("Index is out of range")
    for i in jnp.nonzero(amplitude)[0]:
        if states[i][n] == '0':
            new_amplitude[i+cut] += (p/2)*abs(amplitude[i])**2
            new_amplitude[i] += (1-p/2)*abs(amplitude[i])**2
        else:
            new_amplitude[i-cut] += (p/2)*abs(amplitude[i])**2
            new_amplitude[i] += (1-p/2)*abs(amplitude[i])**2
    #     wavefunction.wave.iloc[0, :] = jnp.sqrt(new_amplitude)
    for i in range(2**qubit_num):
        if amplitude[i] < 0:
            new_amplitude[i] = - jnp.sqrt(new_amplitude[i])
        else:
            new_amplitude[i] = jnp.sqrt(new_amplitude[i])
    wavefunction.amplitude = new_amplitude

def E_all(wavefunction, p_noise, qubit_num):
    if p_noise > 0:
        for i in range(qubit_num):
            E(wavefunction, p_noise, i)