//! Main `Model` trait for expanding this library to work with other models. Leaky integrate and fire is built in.



use std::fmt::Debug;
use self::lif::LifNeuron;


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

}


