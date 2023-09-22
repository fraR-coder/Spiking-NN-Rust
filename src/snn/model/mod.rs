//! Main `Model` trait for expanding this library to work with other models. Leaky integrate and fire is built in.

pub mod lif;
pub trait Model {

    type Neuron: 'static + Sized + Clone ;
     /// Helper type to build neurons
    type Config: ;

    fn handle_spike(neuron: &mut Self::Neuron, weighted_input_val: f64, ts: u128) -> f64;
}