# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:24:01 2021

@author: ASUS
"""
import jax.numpy as jnp
from typing import NamedTuple
from flax import struct
import jax
from dataclasses import dataclass

@jax.tree_util.register_pytree_node_class
@dataclass
class Wavefunction:
    state: jnp.ndarray = struct.field(pytree_node=True)
    amplitude: jnp.ndarray = struct.field(pytree_node=True)
    n_qubits: int = struct.field(pytree_node=True)
    # def __init__(self, state, amplitude, n_qubits=1):
    #     self.state = jnp.array(state)
    #     self.amplitude = jnp.array(amplitude)
    #     self.n_qubits = n_qubits
    
    def probabilities(self):
        """returns a dictionary of associated probabilities."""
        #jax.debug.print("Probabilities: {}", jnp.abs(self.amplitude))
        return jnp.abs(self.amplitude) ** 2

    def print_state(self):
        """represent a quantum state in bra-ket notations"""
        states = self.state
        
        for i in range(0, len(states)):
            jax.debug.print("State {}: Amplitude {}", states[i], self.amplitude[i]) 

    def tree_flatten( self):
        children = (self.state, self.amplitude)
        aux = self.n_qubits
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        state, amplitude = children
        return cls(state, amplitude, aux)
    

    # def visual_circuit(self):
    #     """Visualization of a ciruict"""
    #     n = len((self.state)[0])
    #     a = self.visual
    #     b = [[]]*(2*n)
    #     for i in range(2*n):
    #         b[i] = [0]*len(a)

    #     for i in range(n):
    #         for j in range(len(a)):
    #             if i in a[j]:    
    #                 if ('RX' in a[j]) or ('RY' in a[j]) or ('RZ' in a[j]):
    #                     b[2*i][j] = 1.5
    #                 elif ('CRX' in a[j]) or ('CRY' in a[j]) or ('CRZ' in a[j]):
    #                     b[2*i][j] = 2.5
    #                 elif ('CX' in a[j]) or ('SWAP' in a[j]):
    #                     b[2*i][j] = 3
    #                 elif ('CP' in a[j]):
    #                     b[2*i][j] = 3.5
    #                 elif ('CCX' in a[j]):
    #                     b[2*i][j] = 4
    #                 elif ('CSWAP' in a[j]):
    #                     b[2*i][j] = 5
    #                 else:
    #                     b[2*i][j] = 1

    #     for j in range(len(a)):
    #         if ('CX' in a[j]) or ('CCX' in a[j]) or ('SWAP' in a[j]) or ('CSWAP' in a[j]):
    #             for i in range(2*min(a[j][:-1])+1, 2*max(a[j][:-1]), 2):
    #                 b[i][j] = 2
    #         if ('CP' in a[j]) or ('CRX' in a[j]):
    #             for i in range(2*min(a[j][:-2])+1, 2*max(a[j][:-2]), 2):
    #                 b[i][j] = 2

    #     string_out = [[]]*(2*n)
    #     for i in range(2*n):
    #         string_out[i] = []

    #     for i in range(n):
    #         out = ''
    #         if i < 10:
    #             out += '|Q_'+str(i)+'> : '
    #         else:
    #             out += '|Q_'+str(i)+'>: '
    #         space = ' '*len(out)
    #         string_out[2*i].append(out)
    #         string_out[2*i+1].append(space)

    #         out = ''
    #         space = ''
    #         for j in range(len(a)):

    #             if b[2*i][j] == 0:
    #                 out += '---'

    #             if b[2*i][j] == 1:
    #                 out += a[j][-1] + '--'

    #             if b[2*i][j] == 1.5:
    #                 out += a[j][-2]children, aux + '-'

    #             if b[2*i][j] == 2.5:
    #                 if i == a[j][0]:
    #                     out += 'o--'
    #                 elif i == a[j][1]:
    #                     out += a[j][-2][1:] + '-'

    #             if b[2*i][j] == 3:
    #                 if i == a[j][0]:
    #                     out += 'o--'
    #                 elif i == a[j][1]:
    #                     out += 'x--'

    #             if b[2*i][j] == 3.5:
    #                 if i == a[j][0]:
    #                     out += 'o--'
    #                 elif i == a[j][1]:
    #                     out += a[j][-2][1] + '--'

    #             if b[2*i][j] == 4:
    #                 if i == a[j][0] or i == a[j][1]:
    #                     out += 'o--'
    #                 elif i == a[j][2]:
    #                     out += 'x--'

    #             if b[2*i][j] == 5:
    #                 if i == a[j][0] or i == a[j][1]:
    #                     out += 'x--'
    #                 elif i == a[j][2]:
    #                     out += 'o--'


    #             if b[2*i+1][j] == 2:
    #                 space += '|  '
    #             if b[2*i+1][j] == 0:
    #                 space += '   '

    #         string_out[2*i].append(out+'-M')
    #         string_out[2*i+1].append(space+'  ')

    #     for i in string_out:
    #         print(i[0]+i[1])

def _wf_flatten(wf: Wavefunction):
    children = (wf.state, wf.amplitude)  
    aux = wf.n_qubits                  
    return children, aux

def _wf_unflatten(aux, children):
    state, amplitude = children
    return Wavefunction(state, amplitude, aux)

try:
    jax.tree_util.register_pytree_node(Wavefunction, _wf_flatten, _wf_unflatten)
except ValueError:
    pass