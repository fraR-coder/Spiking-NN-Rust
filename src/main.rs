

pub mod snn;

use crate::snn::model::lif::*;

fn main() {
    let config=Configuration::new(1.0,2.0,3.0,4.0);
    let mut neuron1=LifNeuron::from_conf(&config);
    let neuron2=LifNeuron::from_conf(&config);

    
    let weighted_input_val=10.2;
    let ts=1;


    println!("res: {:?}", handle_spike(&mut neuron1, weighted_input_val, ts));
}
