

pub mod snn;

use crate::snn::model::lif::*;

fn main() {
    let config=Configuration::new(1.0,2.0,3.0,4.0);
    let mut neuron1=LifNeuron::from_conf(&config);
    let neuron2=LifNeuron::from_conf(&config);


    println!("n1: {:?}", neuron1);
}
