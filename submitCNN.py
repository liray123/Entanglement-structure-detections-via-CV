# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:35:13 2024

@author: lr
"""


import numpy as np
import itertools
import functools
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import sparse_categorical_crossentropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def integer_partitions(n, k=None):
    # Generate all integer partitions of 'n' into 'k' parts
    if k is None:
        partitions = set()
        for k in range(1, n + 1):
            for partition in integer_partitions(n, k):
                partitions.add(partition)
        return partitions
    elif k == 1:
        return {(n,)}
    else:
        partitions = set()
        for first in range(1, n // k + 1):
            for subpartition in integer_partitions(n - first, k - 1):
                partitions.add((first,) + subpartition)
        return partitions

def random_probabilities(n):
    # Generate 'n' random probabilities that sum to 1
    probs = np.random.rand(n)
    return probs / np.sum(probs)


def generate_states(num_particles):
    # Generate all possible states for a given number of particles
    partitions = integer_partitions(num_particles)
    
    states = []
    for partition in partitions:
        # Generate all possible permutations for the partition's subsets
        permutations = set(itertools.permutations(partition))
        for perm in permutations:
            state = []
            for num in perm:
                if num == 1:
                    state.append('One')
                elif num == 2:
                    state.append('Bell')
                elif num >= 3:
                    state.append(f'GHZ_{num}')
            states.append(tuple(state))

    # Remove duplicate states
    unique_states = list(set(states))

    return unique_states


def Rx(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]])

def Ry(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2), np.cos(theta/2)]])

def Rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]])

def expand_rotation_matrix(base_rotation, num_qubits):
    expanded_rotation = base_rotation
    for _ in range(1, num_qubits):
        expanded_rotation = np.kron(expanded_rotation, base_rotation)
    return expanded_rotation

def apply_random_rotation_to_subsystem(dm):
    num_qubits = int(np.log2(dm.shape[0]))  # Calculate the number of qubits from the size of the density matrix

    theta_x = np.random.uniform(0, np.pi / 10)
    theta_y = np.random.uniform(0, np.pi / 12)
    theta_z = np.random.uniform(0, np.pi / 10)

    # Generate single-qubit rotation matrices
    Rx_single = Rx(theta_x)
    Ry_single = Ry(theta_y)
    Rz_single = Rz(theta_z)

    # Initialize full rotation matrices as identity matrices
    Rx_full = np.eye(2)
    Ry_full = np.eye(2)
    Rz_full = np.eye(2)

    # Expand the rotation matrices to the size of the subsystem
    for _ in range(1, num_qubits):
        Rx_full = np.kron(Rx_full, Rx_single)
        Ry_full = np.kron(Ry_full, Ry_single)
        Rz_full = np.kron(Rz_full, Rz_single)

    # Apply rotations
    rotation = np.dot(np.dot(Rz_full, Ry_full), Rx_full)
    rotated_dm = np.dot(np.dot(rotation, dm), rotation.conj().T)
    
    return rotated_dm



def generate_density_matrices(states):
    density_matrices = []
    for state in states:
        product_states_dm = []
        for subsystem in state:
            if subsystem == 'One':
                dm = np.array([[0, 0], [0, 1]], dtype=np.float64)
                rotated_dm = apply_random_rotation_to_subsystem(dm)
                product_states_dm.append(rotated_dm)
            elif subsystem == 'Bell':
                dm = np.array([[0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0], [0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5]], dtype=np.float64)
                rotated_dm = apply_random_rotation_to_subsystem(dm)
                product_states_dm.append(rotated_dm)
            elif subsystem.startswith('GHZ'):
                num_particles = int(subsystem.split('_')[1])
                matrix_size = 2 ** num_particles
                dm = np.zeros((matrix_size, matrix_size), dtype=np.float64)
                dm[0, 0] = 0.5
                dm[-1, -1] = 0.5
                rotated_dm = apply_random_rotation_to_subsystem(dm)
                product_states_dm.append(rotated_dm)

        combined_dm = functools.reduce(np.kron, product_states_dm)
        density_matrices.append(combined_dm)
    
    return density_matrices


def stratified_sampling(states, num_samples):
  
    num_classes = len(states)
    samples_per_class = num_samples // num_classes

    sampled_states = []
    sampled_density_matrices = []

    for i in range(num_classes):
        for _ in range(samples_per_class):
            sampled_states.append(states[i])
            density_matrix = generate_density_matrices([states[i]])[0]
            sampled_density_matrices.append(density_matrix)


    combined = list(zip(sampled_states, sampled_density_matrices))
    random.shuffle(combined)
    sampled_states, sampled_density_matrices = zip(*combined)

    return list(sampled_states), list(sampled_density_matrices)



def add_noise(density_matrix, p):

    matrix_size = density_matrix.shape[0]


    I_matrix = np.eye(matrix_size) / matrix_size


    noisy_density_matrix = p * density_matrix + (1 - p) * I_matrix

    return noisy_density_matrix

num_particles = 8

num_samples = 10000

states = generate_states(num_particles)
density_matrices = generate_density_matrices(states)


sampled_states, sampled_density_matrices = stratified_sampling(states, num_samples)


# Add noise to each density matrix
noisy_density_matrices = []
for dm in sampled_density_matrices:
    # Randomly choose a value 'p' in the range [0.9, 1]
    p = np.random.uniform(0.9, 1)

    # Add noise
    noisy_dm = add_noise(dm, p)

    noisy_density_matrices.append(noisy_dm)

# Replace the original density matrices with the noisy ones
sampled_density_matrices = noisy_density_matrices

# Prepare the dataset
X = np.array(sampled_density_matrices)
y = ['-'.join(state) for state in sampled_states]

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the dataset for the CNN
X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(-1, X_val.shape[1], X_val.shape[2], 1)

num_classes = len(np.unique(y))

# Create the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1), strides=(2, 2), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128,kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
    ])

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with training data and validation data
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f'Test accuracy: {val_accuracy}')

