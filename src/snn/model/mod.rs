//! Main `Model` trait for expanding this library to work with other models. Leaky integrate and fire is built in.

use crate::snn::layer::Layer;
use crate::snn::Spike;

use self::lif::LifNeuron;

pub mod lif;

pub trait Model {

    type Neuron: 'static + Sized + Clone + Send + Sync ;
     /// Helper type to build neurons
    type Config: Clone;

    fn handle_spike(neuron: &mut Self::Neuron, weighted_input_val: f64, ts: u128) -> f64;

    
}