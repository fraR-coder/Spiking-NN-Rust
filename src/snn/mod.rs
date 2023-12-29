//! # Spiking Neural Network Library
//!
//! This Rust library provides components and utilities for building and working with spiking neural networks (SNNs).
//! It includes modules for layers, models, neural networks, resilience mechanisms, JSON adapters, and a console input-based NN creator.
//!
//! ## Spike Struct
//!
//! The `Spike` struct represents a spike event in an SNN. It includes timestamp (`ts`), layer index (`layer_id`), and neuron index (`neuron_id`).
//!
//! ```rust
//! #[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
//! pub struct Spike {
//!     pub ts: u128,
//!     pub layer_id: usize,
//!     pub neuron_id: usize,
//! }
//! ```
//!
//! The library provides methods for creating new spikes and merging spikes from a matrix into a single vector.
//!
//! ## HeapCalculator and Link
//!
//! The `HeapCalculator` and `Link` structs offer functionality for performing heap-based calculations in SNNs.
//!
//! ```rust
//! pub struct HeapCalculator<T: Clone, U> {
//!     heap_vec: Vec<Link<T, U>>,
//! }
//!
//! pub struct Link<T: Clone, U> {
//!     value: T,
//!     stuck_bit: Option<Stuck>,
//!     mask: Option<U>,
//!     marker: PhantomData<U>,
//! }
//! ```
//!
//! ## Configuration Handling
//!
//! The library includes modules for handling configurations, such as `ConfigurationJson` for reading configurations from files.
//!
//! ## NeuronJson and Network Initialization
//!
//! The `NeuronJson` module provides functionality to initialize an SNN based on JSON configurations, layers, and weights.
//!
//! ```rust
//! pub struct NeuronJson;
//!
//! impl NeuronJson {
//!     pub fn read_from_file(layers_pathname: &str, weights_pathname: &str, configurations_pathname: &str) -> Result<NN<LeakyIntegrateFire>, String> {
//!         // ... (function details)
//!     }
//! }
//! ```
//!
//! ## Resilience Handling
//!
//! The library includes a `ResilienceJson` module for reading resilience configurations from files and converting them into resilience objects.
//!
//! ```rust
//! pub struct ResilienceJson;
//!
//! impl ResilienceJson {
//!     pub fn read_from_file(pathname: &str) -> Result<ResilienceJson, String> {
//!         // ... (function details)
//!     }
//!
//!     pub fn to_resilience(self) -> Result<Resilience, String> {
//!         // ... (function details)
//!     }
//! }
//! ```
//!
//! ## JSON Adapter
//!
//! The `json_adapter` module provides utilities for reading and handling JSON files related to SNN configurations.
//!
//! ## Console Input Neural Network Creator
//!
//! The `console_input_nn_creator` module offers a mechanism to create an SNN interactively through the console.
//!

use std::fmt;

pub mod layer;
pub mod model;
pub mod nn;
pub mod resilience;

pub mod console_input_nn_creator;
pub mod json_adapter;

/// Represents a spike event in a spiking neural network.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
pub struct Spike {
    /// Timestamp of when the spike occurs.
    pub ts: u128,
    /// Index of the layer, the neuron of which generated the spike.
    pub layer_id: usize,
    /// Index of the neuron which generated the spike.
    pub neuron_id: usize,
}

impl Spike {
    /// Create a new spike at time `ts` for neuron `neuron_id`.
    pub fn new(ts: u128, layer_id: usize, neuron_id: usize) -> Spike {
        Spike {
            ts,
            layer_id,
            neuron_id,
        }
    }

    /// Receive a matrix where each line is a vector of Spikes and merge all the Spikes into a single vector.
    pub fn vec_of_all_spikes(spikes: Vec<(u128, Vec<u128>)>) -> Vec<Spike> {
        let mut res: Vec<Spike> = spikes
            .into_iter()
            .flat_map(|(neuron_id, spikes_vector)| {
                spikes_vector.into_iter().map(move |ts| Spike {
                    ts,
                    layer_id: 0,
                    neuron_id: neuron_id as usize,
                })
            })
            .collect();

        res.sort(); // ascending order by default because of the `Ord` trait
        res
    }
}

impl fmt::Display for Spike {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Spike(ts: {}, layer_id: {}, neuron_id: {})",
            self.ts, self.layer_id, self.neuron_id
        )
    }
}
