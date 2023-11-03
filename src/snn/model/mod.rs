//! Main `Model` trait for expanding this library to work with other models. Leaky integrate and fire is built in.
use std::ops::{Add, BitAndAssign, BitOrAssign, Mul, Not};


use std::fmt::Debug;

pub mod heap;
pub mod lif;
pub mod logic_circuit;

pub trait Model {

    type Neuron: 'static + Sized + Clone + Send + Sync + Debug ;
     /// Helper type to build neurons
    type Config: Clone;
    fn handle_spike(neuron: &mut Self::Neuron, weighted_input_val: f64, ts: u128) -> f64;
    fn update_v_mem(neuron: &mut Self::Neuron, val: f64);

    fn update_v_rest(neuron: &mut Self::Neuron, stuck: bool);
    fn update_v_reset(neuron: &mut Self::Neuron, stuck: bool);
    fn update_v_th(neuron: &mut Self::Neuron, stuck: bool);
    fn update_tau(neuron: &mut Self::Neuron, stuck: bool);

    fn use_full_adder(neuron: &mut Self::Neuron,stuck: bool,n_inputs: usize);

}


// Define the ToBits trait
pub trait ToBits<U> {
    fn get_bits(&self) -> U;
    fn from_bits(bits: U) -> Self;

    fn create_mask(&self, index: i32) -> U; //just to create a mask of bits for different types
                                            //generate a bit sequence with all 0's and a 1 in the position specified by index.
                                            //The lenght depends on the type implementing this

    // Get the number of bits in the type implementing this trait.
    fn num_bits(&self) -> i32;
}

impl ToBits<u64> for f64 {
    // to conversion with bits
    fn get_bits(&self) -> u64 {
        self.to_bits()
    }
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }

    fn create_mask(&self, index: i32) -> u64 {
        1u64 << index
    }

    fn num_bits(&self) -> i32 {
        64 //Modo migliore????
    }
}
//**TODO can be implemented for all the types we need */
/*
The LogicCircuit trait represents a generic logic circuit that performs operations and provides methods
for setting and getting inputs, outputs, and error selectors.
It is parameterized with two types, T and U. T is the type of the values taht perform the operations. U is the type of the rappresentation in bit of T (e.g. T:f64->U:u64)
The error selector is a field to keep track of the selected field( i1,i2,o) and selected bit to apply the error bit injection
*/
pub trait LogicCircuit<T: Add<Output = T> + Mul<Output = T> + Clone, U> {
    fn operation(&mut self, stuck: bool) -> T;
    fn set_random_bit(&mut self,indexes:(i32,i32), stuck: bool);
    fn get_input1(&self) -> T;
    fn set_input1(&mut self, value: T);
    fn get_input2(&self) -> T;
    fn set_input2(&mut self, value: T);
    fn get_output(&self) -> T;
    fn set_output(&mut self, value: T);
    fn get_error_selector(&self) -> Option<(i32, i32)>;
    fn set_error_selector(&mut self, value: Option<(i32, i32)>);
}




