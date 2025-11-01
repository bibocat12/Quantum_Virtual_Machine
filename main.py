from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wandb


from Qsun_jax import Qcircuit
from Qsun_jax import Qgates
from Qsun_jax import Qmeas
from Qsun_jax import QBGate
from Qsun_jax import Qwave

import torch

import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml

import torchsummary
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Sample code is from Quantum Neural Network-MNIST.ipynb
import random
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import time

import torch.nn.functional as F

import os
from tqdm import tqdm

RANDOM_SEED = 42
BATCH_SIZE = 64
def init_random_gates(n_qubit, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)

    targets = jax.random.randint(k1, (n_qubit,), 0, n_qubit)
    gate_ids = jax.random.randint(k2, (n_qubit,), 0, 3)
    cnot_pairs = []

    m = n_qubit // 4
    for k in range(m):
        subkey = jax.random.fold_in(k3, k)
        perm = jax.random.permutation(subkey, n_qubit)
        cnot_pairs.append((perm[0], perm[1]))

    cnot_pairs = jnp.array(cnot_pairs)
    return targets, gate_ids, cnot_pairs

TARGETS, GATE_IDS, CNOT_PAIRS = init_random_gates(9, seed=RANDOM_SEED)

@jax.jit
def layer(circuit, params):
    circuit = Qgates.RY(circuit, 2, params[0])
    circuit = Qgates.RZ(circuit, 5, params[1])
    circuit = Qgates.RX(circuit, 7, params[2])
    circuit = Qgates.RY(circuit, 0, params[3])
    circuit = Qgates.CNOT(circuit, 3, 4)
    circuit = Qgates.CNOT(circuit, 1, 8)
    circuit = Qgates.RX(circuit, 6, params[4])
    circuit = Qgates.RZ(circuit, 3, params[5])
    circuit = Qgates.RY(circuit, 4, params[6])
    circuit = Qgates.RX(circuit, 2, params[7])
    circuit = Qgates.RY(circuit, 5, params[8])
    circuit = Qgates.CNOT(circuit, 6, 2)
    return circuit



# @jax.jit
# def layer(circuit, params):

#     n_qubit = params.shape[0]

#     key = jax.random.PRNGKey(RANDOM_SEED)
#     key, k1, k2, k3 = jax.random.split(key, 4)

#     # Precompute random targets and rotation gate ids (0: RY, 1: RZ, 2: RX)
#     targets = jax.random.randint(k1, (n_qubit,), 0, n_qubit)
#     gate_ids = jax.random.randint(k2, (n_qubit,), 0, 3)

#     def rot_body(i, circ):
#         t = targets[i]
#         gid = gate_ids[i]
#         theta = params[i]
#         def do_ry(c): return Qgates.RY(c, t, theta)
#         def do_rz(c): return Qgates.RZ(c, t, theta)
#         def do_rx(c): return Qgates.RX(c, t, theta)
#         return lax.switch(gid, (do_ry, do_rz, do_rx), circ)

#     circuit = lax.fori_loop(0, n_qubit, rot_body, circuit)

#     # Random CNOT block: m pairs
#     m = n_qubit // 4
#     def cnot_body(k, circ):
#         subkey = jax.random.fold_in(k3, k)
#         perm = jax.random.permutation(subkey, n_qubit)
#         control = perm[0]
#         target = perm[1]
#         return Qgates.CNOT(circ, control, target)

#     circuit = lax.fori_loop(0, m, cnot_body, circuit)
#     return circuit

@jax.jit
def initial_state(sample):
    """
        Encode classical data into quantum states using RY gates.
        Args:
            sample (1D array): Input features to be encoded. (0 <= x_i <= 1)
        Returns:
            circuit_initial (Qcircuit): Quantum circuit with encoded states.
    """
    n_qubits = sample.shape[-1]
    circuit_initial = Qcircuit.Qubit(n_qubits)
    def apply_gate(circ, i):
        angle = lax.cond(sample[i] > 0.5,
                         lambda _: jnp.pi,
                         lambda _: 0.0,
                         operand=None)
        circ = Qgates.RY(circ, i, angle)
        return circ

    # dùng scan thay vì for để hợp lệ JAX hơn nữa
    def body_fun(i, circ):
        circ = apply_gate(circ, i)
        return circ

    # Thực thi vòng lặp trên số qubit
    circuit_model = lax.fori_loop(0, n_qubits, body_fun, circuit_initial)
    return circuit_model

# QNN model to return expectation values
@jax.jit
def qnn_model(input, params):
    """ 
        VQC for Quanvolutional Neural Network
        Args:
            input (1D array): Input features for the QNN. (0 <= x_i <= 1)
            params (2D array): Parameters for the QNN layers.

        Returns:
            q_meas (1D array): Expectation values from the QNN measurement.
    """
   # print("Building the QNN model...")
    input = input.reshape(-1)
    circuit_model = initial_state(input)
    circuit_model = layer(circuit_model, params)
    
    # Use JAX functional loop for measurements
    def measure_qubit(i, q_meas_array):
        prob = Qmeas.measure_one(circuit_model, i)
        expectation = prob[0] * (-1) + prob[1] * 1
        q_meas_array = q_meas_array.at[i].set(expectation)
        return q_meas_array
    
    # Initialize array and loop over qubits
    n_qubits = len(input)
    q_meas_init = jnp.zeros(n_qubits, dtype=jnp.float32)
    q_meas = lax.fori_loop(0, n_qubits, measure_qubit, q_meas_init)
    
    return q_meas

# QNN model to return expectation values
@jax.jit
def batch_qnn_model(input, params):
    """ 
        VQC for Quanvolutional Neural Network
        Args:
            input (1D array): Input features for the QNN. (0 <= x_i <= 1)
            params (2D array): Parameters for the QNN layers.

        Returns:
            q_meas (1D array): Expectation values from the QNN measurement.
    """

    sample_jax = jnp.array(input)
    # params to length of input.shape[0]
    params_jax = jnp.array(params)

    batched_results = jax.vmap(qnn_model)(sample_jax, params_jax)

    return batched_results


def load_data():
    train_dataset = datasets.MNIST("./", train=True, download=True)
    test_dataset = datasets.MNIST("./", train=False, download=True)

    # Convert to NumPy arrays
    X_train = np.array([np.array(img) for img in train_dataset.data])  
    y_train = np.array(train_dataset.targets)  
    X_test = np.array([np.array(img) for img in test_dataset.data])  
    y_test = np.array(test_dataset.targets) 


    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    print(f"Image shape: {X_train[0].shape}")

    return X_train, y_train, X_test, y_test

class quanKernelBatch(torch.autograd.Function):
    """
    Forward:
      input:  (B, patch_size) torch tensor
      params: (n_params,) torch tensor (shared params)
    Output:
      (B, vec_len) torch tensor
    """
    @staticmethod
    def forward(ctx, input, params):

        device = input.device

        outputs_jax = batch_qnn_model(input, params)       # JAX DeviceArray (B, vec_len)

        # Convert JAX -> host NumPy for PyTorch
        outputs_np = jax.device_get(outputs_jax)
        outputs_np = np.asarray(outputs_np, dtype=np.float32)

        # Lưu để backward
        ctx.save_for_backward(input, params)
        ctx.vec_len = outputs_np.shape[1]

        out_tensor = torch.from_numpy(outputs_np)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (B, vec_len)
        # input, params = ctx.saved_tensors
        # device = grad_output.device

        # input_np = input.detach().cpu().numpy()    # (B, patch_size)
        # params_np = params.detach().cpu().numpy()  # (n_params,)

        # B = input_np.shape[0]
        # n_params = params_np.shape[0]
        # vec_len = ctx.vec_len

        # grad_params = np.zeros_like(params_np, dtype=np.float32)

        # # parameter-shift cho từng param, tính cho toàn bộ batch + tất cả vector outputs
        # for k in range(n_params):
        #     shifted_plus = params_np.copy()
        #     shifted_plus[k] += np.pi/2
        #     f_plus = np.stack([
        #         np.asarray(qnn_model(input_np[i], shifted_plus)).astype(np.float32).reshape(-1)
        #         for i in range(B)
        #     ], axis=0)  # (B, vec_len)

        #     shifted_minus = params_np.copy()
        #     shifted_minus[k] -= np.pi/2
        #     f_minus = np.stack([
        #         np.asarray(qnn_model(input_np[i], shifted_minus)).astype(np.float32).reshape(-1)
        #         for i in range(B)
        #     ], axis=0)  # (B, vec_len)

        #     # grad_output (B, vec_len)  -> multiply elementwise then sum
        #     grad_out_np = grad_output.detach().cpu().numpy()  # (B, vec_len)
        #     diff = (f_plus - f_minus)  # (B, vec_len)
        #     grad_params[k] = np.sum(diff * grad_out_np) / 2.0

        # grad_params_tensor = torch.from_numpy(grad_params).to(device).float()
        # grad_input = None  # không tính gradient w.r.t input patch
        return None, None
    
# class QuanConv2Module(torch.nn.Module):
#     def __init__(self, input_size, output_size=1, kernel_size=3, stride=2, padding=0, fixed_params=None):
#         super().__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

#         if fixed_params is not None and len(fixed_params) > 0:
#             self.params = torch.nn.Parameter(torch.tensor(fixed_params, dtype=torch.float32), requires_grad=False)
#         else:
#             # shape: (in_channels, output_size, kernel_size*kernel_size)
#             self.params = torch.nn.Parameter(
#                 torch.randn(input_size, output_size, kernel_size * kernel_size) * np.pi
#             )

#     def forward(self, x):
#         """
#         x: (B, C, H, W)
#         returns: (B, output_size * vec_len, out_h, out_w)
#         where vec_len is length of vector returned by quanKernel (e.g. kernel_size*kernel_size)
#         """
#         B, C, H, W = x.shape
#         k = self.kernel_size
#         s = self.stride
#         p = self.padding

#         out_h = (H - k + 2*p) // s + 1
#         out_w = (W - k + 2*p) // s + 1

#         if p > 0:
#             x = F.pad(x, (p, p, p, p))


#         patches = F.unfold(x, kernel_size=k, stride=s) 
#         patches = patches.transpose(1, 2)           
#         B, P, patch_size = patches.shape
#         patches_flat = patches.reshape(B * P, patch_size) 

#         device = x.device
#         outputs_per_oc = []
#         for oc in range(self.output_size):

#             params_oc = self.params[:, oc, :].reshape(-1).detach()

#             res_flat = quanKernelBatch.apply(patches_flat.to(device), params_oc.to(device)) 

#             res_bp = res_flat.reshape(B, P, -1)
#             outputs_per_oc.append(res_bp)


#         stacked = torch.stack(outputs_per_oc, dim=0).permute(1, 0, 2, 3)  # (B, output_size, P, vec_len)


#         B, O, P, V = stacked.shape
#         stacked = stacked.reshape(B, O * V, P)  
#         out = stacked.view(B, O * V, out_h, out_w)
#         return out

class quanNN(torch.nn.Module):
    def __init__(self):
        super(quanNN, self).__init__()
        # self.quanConv2D = QuanConv2Module(input_size=1, output_size=2, 
        #     kernel_size=3, 
        #     stride=1, 
        #     padding=0, 
        #     fixed_params=random_params)

        self.conv2d = torch.nn.Conv2d(in_channels=9, out_channels=8, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, padding=2)
        self.relu = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(144, 128)  # Adjust input features based on output size after conv and pooling
        self.fc2 = torch.nn.Linear(128, 10)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        # x = self.quanConv2D(x)

        x = self.conv2d(x)  # Add batch dimension for conv2d
        x = self.relu(x)
        x = self.pool1(x)  # Add batch dimension for pooling
        x = self.conv2d2(x)
        x = self.relu(x)
        x = self.pool2(x)  # Remove batch dimension after pooling
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def conv2d_q(x, input_size, output_size=1, kernel_size=3, stride=1, padding=0, fixed_params=None):


    B, C, H, W = x.shape
    out_h = (H - kernel_size + 2*padding) // stride + 1
    out_w = (W - kernel_size + 2*padding) // stride + 1

    start_time = time.time()
    patches = []
    paramss = []
    for b in range(B):
        for oc in range(output_size):
            for i in range(out_h):
                for j in range(out_w):
                    region = x[b, :, i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
                    patches.append(region)
                    paramss.append(fixed_params[:, oc, :].reshape(-1))
    end_time = time.time()

    #print(fixed_params.shape, jax.numpy.array(patches).shape)
    res_flat = quanKernelBatch.apply(jax.numpy.stack(patches), jax.numpy.array(paramss))
    res_bp = res_flat.reshape(B, output_size * kernel_size * kernel_size, out_h, out_w)
    return res_bp

def dataloader_q(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE, shuffle=True):
    x_train_quantum = []
    np.random.seed(RANDOM_SEED)
    random_params = np.random.rand(1, 1, 9) * np.pi 
    print("Random fixed parameters shape:", random_params.shape)
    # QuanConv2_begin = QuanConv2Module(input_size=1, output_size=2, kernel_size=3, stride=1, padding=0, fixed_params=random_params)

    quantum_outputs_path = "./output2/"
    os.makedirs(quantum_outputs_path, exist_ok=True)
    for start in tqdm(range(0, x_train.shape[0], batch_size)):
        end = min(start + batch_size, x_train.shape[0])
        batch_path = f"{quantum_outputs_path}/x_train_quantum_{start}_{end}.pt"

        if os.path.exists(batch_path):
            x_train_quantum.append(torch.load(batch_path))
            continue
        start_time = time.time()
        batch = x_train[start:end].reshape(-1, 1, 28, 28) / 255.0
        outputs = conv2d_q(batch, input_size=1, output_size=1, kernel_size=3, stride=2, padding=0, fixed_params=random_params)

        if isinstance(outputs, torch.Tensor):
            outputs_torch = outputs
        else:
            outputs_torch = torch.stack(outputs)
        torch.save(outputs_torch, batch_path)
        x_train_quantum.append(outputs_torch)
        end_time = time.time()
        print(f"Processed batch {start}-{end} in {end_time - start_time:.2f} seconds.")

    x_train_quantum_tensor = torch.cat(x_train_quantum, dim=0)
    y_train = [int(label) for label in y_train]
    y_train_tensor = torch.tensor(y_train)


    train_loader_q = DataLoader(
        TensorDataset(x_train_quantum_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    x_test_quantum = []
    os.makedirs(quantum_outputs_path, exist_ok=True)
    for start in tqdm(range(0, x_test.shape[0], BATCH_SIZE)):
        end = min(start + BATCH_SIZE, x_test.shape[0])
        batch_path = f"{quantum_outputs_path}/x_test_quantum_{start}_{end}.pt"

        if os.path.exists(batch_path):
            x_test_quantum.append(torch.load(batch_path))
            continue

        batch = x_test[start:end].reshape(-1, 1, 28, 28) / 255.0
        outputs = conv2d_q(batch, input_size=1, output_size=2, kernel_size=3, stride=1, padding=0, fixed_params=random_params)
        if isinstance(outputs, torch.Tensor):
            outputs_torch = outputs
        else:
            outputs_torch = torch.stack(outputs)

        torch.save(outputs_torch, batch_path)
        x_test_quantum.append(outputs_torch)

    x_test_quantum_tensor = torch.cat(x_test_quantum, dim=0)
    y_test = [int(label) for label in y_test]
    y_test_tensor = torch.tensor(y_test)

    test_loader_q = DataLoader(
        TensorDataset(x_test_quantum_tensor, y_test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader_q, test_loader_q

def dataloader(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE, shuffle=True):
    X_train_tensor = torch.tensor(x_train).float().reshape(-1, 1, 28, 28)
    X_test_tensor = torch.tensor(x_test).float().reshape(-1, 1, 28, 28)

    y_train = [int(label) for label in y_train]
    y_train_tensor = torch.tensor(y_train)

    y_test = [int(label) for label in y_test]
    y_test_tensor = torch.tensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=batch_size,
        shuffle=False
    )


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()



    quan_model = quanNN()
    print(quan_model.parameters)
    torchsummary.summary(quan_model, (9, 15, 15))

    train_loader_q, test_loader_q = dataloader_q(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    wandb.init(project="quantum-cnn-mnist", name="quan_model_run_cpu_3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quan_model.to(device)

    epochs = 500
    quan_model.train()

    output_path = "./model_output"
    os.makedirs(output_path, exist_ok=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(quan_model.parameters(), lr=0.0001)
    EPOCHS_VAL = 5

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress = tqdm(train_loader_q, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, labels) in enumerate(progress):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = quan_model(images)
            loss = criterion(outputs.to(device), labels)
            loss.backward()

            # Tính Grad Norm
            grad_norm = 0.0
            for p in quan_model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            optimizer.step()

            # Batch Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc_batch = correct / total
            
            running_loss += loss.item()

            wandb.log({
                "batch_loss": loss.item(),
                "batch_accuracy": acc_batch,
                "grad_norm": grad_norm,
                "epoch": epoch + 1
            })

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc_batch:.4f}",
                "grad_norm": f"{grad_norm:.2f}"
            })

        avg_loss = running_loss / len(train_loader_q)
        avg_acc = correct / total

        wandb.log({
            "epoch_loss": avg_loss,
            "epoch_accuracy": avg_acc,
            "epoch_num": epoch + 1
        })

        torch.save(quan_model.state_dict(), f"{output_path}/quan_model_state_{epoch+1}.pth")

        print(f"✅ Epoch {epoch+1}: Loss {avg_loss:.4f}, Acc {avg_acc:.4f}")

        if (epoch + 1) % EPOCHS_VAL == 0:
            quan_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in test_loader_q:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = quan_model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(test_loader_q)
            val_acc = val_correct / val_total

            wandb.log({
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_epoch": epoch + 1
            })

            print(f"🔍 Validation {epoch+1}: Loss {val_loss:.4f}, Acc {val_acc:.4f}")

            quan_model.train()

    wandb.finish()
    print("🎯 Training completed with wandb logging + validation!")

