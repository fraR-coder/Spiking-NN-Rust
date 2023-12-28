//! The Leaky Integrate and Fire (LIF) module for Spiking Neural Networks.
//!
//! The LIF module contains implementations for the Leaky Integrate and Fire neuron model,
//! which is commonly used in Spiking Neural Networks (SNNs). The `LifNeuron` struct represents
//! an individual LIF neuron, and the `Configuration` struct provides a convenient way to
//! create a specific configuration for LIF neurons.
//!
//! Additionally, this module defines the `InjectionStruct` struct, which represents an injection
//! of a specific stuck value at a given index in the membrane potential of a neuron.
//!
//! The `LeakyIntegrateFire` struct implements the `Model` trait for the LIF neuron model.
//! This trait defines the behavior and characteristics of a neuron model used in an SNN.
//!
//! Example:
//! ```
//! use your_crate_name::snn::lif::{LeakyIntegrateFire, LifNeuron, Configuration};
//!
//! // Create a specific configuration for LIF neurons
//! let lif_config = Configuration::new(-65.0, -70.0, -55.0, 20.0);
//!
//! // Create a new LIF neuron using the configuration
//! let mut lif_neuron = LifNeuron::from_conf(&lif_config);
//!
//! // Handle a spike event for the neuron
//! let output = LeakyIntegrateFire::handle_spike(&mut lif_neuron, 10.0, 100);
//!
//! // Update the membrane potential of the neuron
//! LeakyIntegrateFire::update_v_mem(&mut lif_neuron, 5.0);
//!
//! // Check if the neuron has generated a new spike
//! if output > 0.0 {
//!     println!("Neuron fired!");
//! }
//! ```

use crate::snn::model::Model;

use rand::Rng;
use std::f64;

use super::{heap::HeapCalculator, Stuck, Model};

/// Struct representing an injection of a specific stuck value at a given index in the membrane potential.
#[derive(Clone, Debug)]
pub struct InjectionStruct {
    stuck: Stuck,
    index: usize,
}

/// Struct representing a Leaky Integrate and Fire (LIF) neuron.
#[derive(Clone, Debug)]
pub struct LifNeuron {
    /// Rest potential
    pub v_rest: f64,
    /// Reset potential
    pub v_reset: f64,
    /// Threshold potential
    pub v_th: f64,
    /// Membrane's time constant. This is the product of its capacity and resistance
    pub tau: f64,

    pub v_mem: f64,
    pub ts_old: u128,

    /// Heap vector used to compute the sum when there is a bit error injection in a full adder
    pub heap_tree: Option<HeapCalculator<f64, u64>>,
    /// struct to store information used to update the v_mem of the neuron when there is an error injection
    pub injection_vmem: Option<InjectionStruct>,
}

/// Struct representing a specific configuration for LIF neurons.
#[derive(Clone, Debug)]
pub struct Configuration {
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64,
}

// IMPLEMENTATION FOR LIF NEURONS & LIF NEURON CONFIG

impl LifNeuron {
    pub fn new(v_rest: f64, v_reset: f64, v_th: f64, tau: f64) -> LifNeuron {
        LifNeuron {
            // parameters
            v_rest,
            v_reset,
            v_th,
            tau,
            v_mem: 0.0, //inizialmente a 0?
            ts_old: 0,

            heap_tree: None,
            injection_vmem: None,
        }
    }

    pub fn from_conf(nc: &Configuration) -> LifNeuron {
        Self::new(nc.v_rest, nc.v_reset, nc.v_threshold, nc.tau)
    }

    pub fn update_vmem(&mut self, val: f64) {
        self.v_mem += val;
    }

    /// Create a new array of n [LifNeuron] structs, starting from a given Configuration.

    pub fn new_vec(conf: Configuration, n: usize) -> Vec<LifNeuron> {
        let mut res: Vec<LifNeuron> = Vec::with_capacity(n);

        for _i in 0..n {
            res.push(LifNeuron::from_conf(&conf));
        }

        res
    }
}
// Implementazione del trait Model per LifNeuron

impl Configuration {
    /// Create a new Configuration, which can be used to build one or more identical neurons.

    pub fn new(v_rest: f64, v_reset: f64, v_threshold: f64, tau: f64) -> Configuration {
        Configuration {
            v_rest,
            v_reset,
            v_threshold,
            tau,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LeakyIntegrateFire;

impl Model for LeakyIntegrateFire {
    type Neuron = LifNeuron;

    type Config = Configuration;

    /// Update the value of current membrane tension.
    /// When the neuron receives one or more impulses, it computes the new tension of the membrane
    ///
    ///

    ///
    /// This neuron receives a spike at time of spike _ts_ from a number of its input synapses.
    /// The overall weighted input value of this spike (i.e. the sum, across every lit up input synapse,
    /// of the weight of that synapse) is provided via the _weighted_input_val_ parameter.
    ///
    /// The output of this function is 1.0 iff the neuron has generated a new spike at time _ts_, or 0.0 otherwise.
    ///
    ///

    fn handle_spike(neuron: &mut LifNeuron, weighted_input_val: f64, ts: u128) -> f64 {
        //apply injection if necessary
        let mut random_bit_index: usize = 0;
        if let Some(injection_vmem) = Self::get_injection_vmem(neuron) {
            random_bit_index = injection_vmem.index;
            let mut bits: u64 = neuron.v_mem.to_bits();
            match injection_vmem.stuck {
                Stuck::Zero => bits &= !(1u64 << random_bit_index),
                Stuck::One => bits |= 1u64 << random_bit_index,
                Stuck::Transient => bits ^= 1u64 << random_bit_index,
            };
            neuron.v_mem = f64::from_bits(bits);
        }

        //do the calculations
        //println!("ts: {}, ts_old: {}",ts, neuron.ts_old);
        let delta_t: f64 = (ts - neuron.ts_old) as f64;
        neuron.ts_old = ts;

        // compute the new v_mem value
        neuron.v_mem = neuron.v_rest
            + (neuron.v_mem - neuron.v_rest) * (-delta_t / neuron.tau).exp()
            + weighted_input_val;

        //apply stuck
        if let Some(injection_vmem) = Self::get_injection_vmem(neuron) {
            let mut bits: u64 = neuron.v_mem.to_bits();
            match injection_vmem.stuck {
                Stuck::Zero => bits &= !(1u64 << random_bit_index),
                Stuck::One => bits |= 1u64 << random_bit_index,
                Stuck::Transient => bits ^= 1u64 << random_bit_index,
            };
            neuron.v_mem = f64::from_bits(bits);
        }

        if neuron.v_mem > neuron.v_th {
            neuron.v_mem = neuron.v_reset;
            1.0
        } else {
            0.0
        }
    }

    fn update_v_mem(neuron: &mut LifNeuron, val: f64) {
        if neuron.v_mem + val >= 0.0 {
            neuron.v_mem += val;
        } else {
            neuron.v_mem = 0.0;
        }
    }

    fn update_v_rest(neuron: &mut Self::Neuron, stuck: Stuck) {
        let mut bits = neuron.v_rest.to_bits();
        //println!("vecchi bit: {}",bits);
        let random_bit_index = rand::thread_rng().gen_range(0..64);
        match stuck {
            Stuck::Zero => bits &= !(1u64 << random_bit_index),
            Stuck::One => bits |= 1u64 << random_bit_index,
            Stuck::Transient => bits ^= 1u64 << random_bit_index,
        };
        //println!("update_v_rest: {}",random_bit_index);
        //println!("nuovi bit: {}",bits);
        neuron.v_rest = f64::from_bits(bits);
    }

    fn update_v_reset(neuron: &mut Self::Neuron, stuck: Stuck) {
        let mut bits: u64 = neuron.v_reset.to_bits();
        let random_bit_index = rand::thread_rng().gen_range(0..64);
        match stuck {
            Stuck::Zero => bits &= !(1u64 << random_bit_index),
            Stuck::One => bits |= 1u64 << random_bit_index,
            Stuck::Transient => bits ^= 1u64 << random_bit_index,
        };
        neuron.v_reset = f64::from_bits(bits);
    }

    fn update_v_th(neuron: &mut Self::Neuron, stuck: Stuck) {
        let mut bits: u64 = neuron.v_th.to_bits();
        let random_bit_index = rand::thread_rng().gen_range(0..64);
        match stuck {
            Stuck::Zero => bits &= !(1u64 << random_bit_index),
            Stuck::One => bits |= 1u64 << random_bit_index,
            Stuck::Transient => bits ^= 1u64 << random_bit_index,
        };

        neuron.v_th = f64::from_bits(bits);
    }

    fn update_tau(neuron: &mut Self::Neuron, stuck: Stuck) {
        let mut bits: u64 = neuron.tau.to_bits();
        let random_bit_index = rand::thread_rng().gen_range(0..64);
        match stuck {
            Stuck::Zero => bits &= !(1u64 << random_bit_index),
            Stuck::One => bits |= 1u64 << random_bit_index,
            Stuck::Transient => bits ^= 1u64 << random_bit_index,
        };
        neuron.tau = f64::from_bits(bits);
    }

    fn use_heap(neuron: &mut Self::Neuron, stuck: Stuck, inputs: Vec<f64>) {
        let dim = (2u32).pow(((inputs.len() as f64).log2().ceil()) as u32) as usize;
        let heap_calculator = HeapCalculator::new(dim, stuck);
        neuron.heap_tree = Some(heap_calculator);
    }

    fn use_v_mem_with_injection(neuron: &mut Self::Neuron, stuck: Stuck) {
        let random_bit_index = rand::thread_rng().gen_range(0..64);
        let injection_vmem = InjectionStruct {
            stuck,
            index: random_bit_index,
        };
        neuron.injection_vmem = Some(injection_vmem);
    }

    fn get_heap(neuron: &Self::Neuron) -> Option<HeapCalculator<f64, u64>> {
        neuron.heap_tree.clone()
    }
    fn get_injection_vmem(neuron: &Self::Neuron) -> Option<InjectionStruct> {
        neuron.injection_vmem.clone()
    }
}
