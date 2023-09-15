//! Implementation of the Leaky Integrate and Fire (LIF) model for Spiking Neural Networks
use crate::Model;

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
    /// Flag to mark bit change
    pub stuck_at_zero: bool,

    pub v_mem: f64,
    pub ts_old: u128,
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
    pub fn new(v_rest: f64, v_reset: f64, v_threshold: f64, tau: f64) -> LifNeuron {
        LifNeuron {
            // parameters
            v_rest,
            v_reset,
            v_threshold,
            tau,
            v_mem: 0.0, //inizlamente a 0?
            ts_old: 0,
        }
    }

    pub fn from_conf(nc: &Configuration) -> LifNeuron {
        Self::new(nc.v_rest, nc.v_reset, nc.v_threshold, nc.tau)
    }
}

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
