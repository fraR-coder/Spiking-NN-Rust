//! Main `Model` trait for expanding this library to work with other models. Leaky integrate and fire is built in.

use std::fmt::Debug;

use self::{heap::HeapCalculator, lif::InjectionStruct};

pub mod heap;
pub mod lif;

/// The main trait for neuron models.
pub trait Model {
    type Neuron: 'static + Sized + Clone + Send + Sync + Debug;

    /// Helper type to build neurons
    type Config: Clone;

    /// Handles a spike event for the neuron.
    ///
    /// # Arguments
    ///
    /// * `neuron` - A mutable reference to the neuron.
    /// * `weighted_input_val` - The overall weighted input value of the spike.
    /// * `ts` - The timestamp of the spike.
    ///
    /// # Returns
    ///
    /// The output of this function is 1.0 if the neuron generates a new spike, or 0.0 otherwise.
    fn handle_spike(neuron: &mut Self::Neuron, weighted_input_val: f64, ts: u128) -> f64;

    /// Updates the membrane potential of the neuron.
    fn update_v_mem(neuron: &mut Self::Neuron, val: f64);

    /// Updates the rest potential of the neuron with a stuck value.
    fn update_v_rest(neuron: &mut Self::Neuron, stuck: Stuck);

    /// Updates the reset potential of the neuron with a stuck value.
    fn update_v_reset(neuron: &mut Self::Neuron, stuck: Stuck);

    /// Updates the threshold potential of the neuron with a stuck value.
    fn update_v_th(neuron: &mut Self::Neuron, stuck: Stuck);

    /// Updates the membrane's time constant of the neuron with a stuck value.
    fn update_tau(neuron: &mut Self::Neuron, stuck: Stuck);

    /// Configures the neuron to use a heap for calculations.
    fn use_heap(neuron: &mut Self::Neuron, stuck: Stuck, inputs: Vec<f64>);

    /// Configures the neuron to use a special injection for membrane potential.
    fn use_v_mem_with_injection(neuron: &mut Self::Neuron, stuck: Stuck);

    /// Retrieves the heap calculator used by the neuron, if available.
    fn get_heap(neuron: &Self::Neuron) -> Option<HeapCalculator<f64, u64>>;

    /// Retrieves the injection structure for the neuron, if available.
    fn get_injection_vmem(neuron: &Self::Neuron) -> Option<InjectionStruct>;
}

/// Enum representing different stuck values.
#[derive(Debug, Clone)]
pub enum Stuck {
    Zero,
    One,
    Transient,
}

/// Trait for converting types to bits.
pub trait ToBits<U> {
    /// Gets the bits representation of the type.
    fn get_bits(&self) -> U;

    /// Creates the type from bits representation.
    fn from_bits(bits: U) -> Self;

    /// Creates a mask with a set bit at the specified index.
    fn create_mask(&self, index: u64) -> U;

    /// Gets the number of bits in the type.
    fn num_bits(&self) -> u64;
}

impl ToBits<u64> for f64 {
    fn get_bits(&self) -> u64 {
        self.to_bits()
    }

    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }

    fn create_mask(&self, index: u64) -> u64 {
        1u64 << index
    }

    fn num_bits(&self) -> u64 {
        64
    }
}
