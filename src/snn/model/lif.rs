//! Implementation of the Leaky Integrate and Fire (LIF) model for Spiking Neural Networks

use crate::snn::model::Model;
use rand::Rng;

use std::f64;

use super::{Stuck, heap::HeapCalculator};

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

    pub heap_tree: Option<HeapCalculator<f64, u64>>,
}
/// A struct used to create a specific configuration, simply reusable for other neurons

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
        // This early exit serves as a small optimization
        if weighted_input_val == 0.0 {
            return 0.0;
        }
        //println!("ts: {}, ts_old: {}",ts, neuron.ts_old);
        let delta_t: f64 = (ts - neuron.ts_old) as f64;
        neuron.ts_old = ts;

        // compute the new v_mem value
        neuron.v_mem = neuron.v_rest
            + (neuron.v_mem - neuron.v_rest) * (-delta_t / neuron.tau).exp()
            + weighted_input_val;

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
        let dim =(2u32).pow((( inputs.len() as f64).log2().ceil()) as u32) as usize;
        let heap_calculator=HeapCalculator::new(dim, stuck);
        neuron.heap_tree=Some(heap_calculator);
        
    }

    fn get_heap(neuron: &Self::Neuron)->Option<HeapCalculator<f64,u64>> {
        neuron.heap_tree.clone()
    }
}
