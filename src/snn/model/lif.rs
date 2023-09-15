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
    pub stuck_at_zero: bool
}

impl From<&LifNeuron> for LifSolverVars {
    fn from(neuron: &LifNeuron) -> Self {
        Self {
            v_mem: neuron.v_rest,
            ts_old: 0
        }
    }
}