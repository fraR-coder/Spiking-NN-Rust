# Spiking Neural Network Simulator with Resilience Testing

## Overview
This project is a Spiking Neural Network (SNN) simulator implemented in Rust, a high-performance and memory-safe programming language. The primary goal is to provide a flexible and efficient platform for modeling and simulating the behavior of spiking neurons while incorporating resilience testing through error injection.

## Features

### 1. Spiking Neural Network Simulation
- Utilizes Rust's concurrency and performance features to simulate the dynamics of spiking neurons.
- Implements various neuron models, synapse types, and connectivity patterns for comprehensive neural network modeling.

### 2. Resilience Testing
- Incorporates resilience testing mechanisms to evaluate the robustness of the SNN under various error scenarios.
- Allows for controlled injection of errors, such as noise, signal corruption, or connectivity disruptions, to assess the network's ability to recover and adapt.

### 3. Extensible Architecture
- Designed with modularity and extensibility in mind, allowing users to easily integrate custom neuron models, synapse types, and resilience testing strategies.

## Getting Started

### Prerequisites
- [Rust](https://www.rust-lang.org/learn/get-started) installed on your machine.

### Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/spiking-nn-rust.git
    cd spiking-nn-rust
    ```

2. Build the project:

    ```bash
    cargo build 
    ```

### Usage
1. Run the simulator:

    ```bash
    cargo run 
    ```

2. Explore the configuration files to customize the neural network parameters and resilience testing scenarios.
3. Also a command line configuration of the parameters can be done.

## Resilience Testing
- To conduct resilience tests, modify the error injection parameters in the configuration files.
- Run simulations with varying error conditions to assess the SNN's performance under stress.


