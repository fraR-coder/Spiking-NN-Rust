

pub mod snn;

use crate::snn::model::{lif::*,Model};

use nalgebra::DMatrix;
use snn::layer::*;
use snn::Spike;

use crate::snn::nn::NN;
fn main() {
    let config=Configuration::new(1.0,2.0,3.0,4.0);
    let mut neuron1=LifNeuron::from_conf(&config);
    let neuron2=LifNeuron::from_conf(&config);


    let weighted_input_val=10.2;
    let ts=1;


    let input_weights=DMatrix::from_vec(2, 2, vec![1.0,0.0,0.0,2.0]);
    let intra_weights=DMatrix::from_vec(2, 2, vec![0.0,3.0,4.0,0.0]);
    let v=LifNeuron::new_vec(config, 2);
    let nn= NN::<LeakyIntegrateFire>::new().layer(v, input_weights, intra_weights);
   

    let le:Layer<LeakyIntegrateFire>=Layer::new(neuron2);

    

    println!("res: {}", LeakyIntegrateFire::handle_spike(&mut neuron1, weighted_input_val, ts));

    
    let s=Spike::new(1,1);

    let s1=Spike::vec_of_spike_for(1, vec![1,3,7]);
    let s2=Spike::vec_of_spike_for(2, vec![2,3,5]);

    let mut vs=Vec::new();
    vs.push(s1);
    vs.push(s2);

    let st=Spike::vec_of_all_spikes(vs);

    
    println!("{:?}", st);
   



    
}
