//! Main `Model` trait for expanding this library to work with other models. Leaky integrate and fire is built in.

use std::fmt::Debug;


use self::{heap::HeapCalculator, lif::InjectionStruct};




pub mod heap;
pub mod lif;


pub trait Model {

    type Neuron: 'static + Sized + Clone + Send + Sync + Debug ;
     /// Helper type to build neurons
    type Config: Clone;
    fn handle_spike(neuron: &mut Self::Neuron, weighted_input_val: f64, ts: u128) -> f64;
    fn update_v_mem(neuron: &mut Self::Neuron, val: f64);

    fn update_v_rest(neuron: &mut Self::Neuron, stuck: Stuck);
    fn update_v_reset(neuron: &mut Self::Neuron, stuck: Stuck);
    fn update_v_th(neuron: &mut Self::Neuron, stuck: Stuck);
    fn update_tau(neuron: &mut Self::Neuron, stuck: Stuck);

    fn use_heap(neuron: &mut Self::Neuron,stuck: Stuck,inputs: Vec<f64>);

    fn use_v_mem_with_injection(neuron: &mut Self::Neuron,stuck: Stuck);


    fn get_heap(neuron: &Self::Neuron)->Option<HeapCalculator<f64,u64>>;
    fn get_injection_vmem(neuron: &Self::Neuron)->Option<InjectionStruct>;


}

#[derive(Debug,Clone)]
pub enum Stuck {
    Zero,
    One,
    Transient,
}

// Define the ToBits trait
pub trait ToBits<U> {
    fn get_bits(&self) -> U;
    fn from_bits(bits: U) -> Self;

    fn create_mask(&self, index: u64) -> U; //just to create a mask of bits for different types
                                            //generate a bit sequence with all 0's and a 1 in the position specified by index.
                                            //The lenght depends on the type implementing this

    // Get the number of bits in the type implementing this trait.
    fn num_bits(&self) -> u64;


}

impl ToBits<u64> for f64 {
    // to conversion with bits
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
        64 //Modo migliore????
    }
}








