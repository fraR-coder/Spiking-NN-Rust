pub mod snn;

use crate::snn::model::{lif::*, Model};

use nalgebra::DMatrix;
use snn::layer::*;
use snn::Spike;

use crate::snn::nn::NN;

fn f1() {
    let config = Configuration::new(1.0, 2.0, 3.0, 4.0);
    let mut neuron1 = LifNeuron::from_conf(&config);
    let neuron2 = LifNeuron::from_conf(&config);

    let weighted_input_val = 10.2;
    let ts = 1;

    let input_weights = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
    let intra_weights = DMatrix::from_vec(2, 2, vec![0.0, 3.0, 4.0, 0.0]);
    let v = LifNeuron::new_vec(config, 2);
    let nn = NN::<LeakyIntegrateFire>::new().layer(v, input_weights, intra_weights);

    let le: Layer<LeakyIntegrateFire> = Layer::new(neuron2);

    println!(
        "res: {}",
        LeakyIntegrateFire::handle_spike(&mut neuron1, weighted_input_val, ts)
    );

    let s = Spike::new(1, 1);

    let s1 = Spike::vec_of_spike_for(1, vec![1, 3, 7]);
    let s2 = Spike::vec_of_spike_for(2, vec![2, 3, 5]);

    let mut vs = Vec::new();
    vs.push(s1);
    vs.push(s2);

    let st = Spike::vec_of_all_spikes(vs);

    println!("{:?}", st);
}

fn f2() -> Result<NN<LeakyIntegrateFire>, String> {
    let config1 = Configuration::new(1.0, 2.0, 3.0, 4.0);
    let config2 = Configuration::new(3.0, 1.0, 5.0, 5.0);

    let mut neuron1 = LifNeuron::from_conf(&config1);
    let neuron2 = LifNeuron::from_conf(&config2);

    let mut v = Vec::new();
    v.push(neuron1);
    v.push(neuron2);

    let input_weights0 = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
    let intra_weights0 = DMatrix::from_vec(2, 2, vec![0.0, 3.0, 4.0, 0.0]);

    let input_weights1 = DMatrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 3.0]);
    let intra_weights1 = DMatrix::from_vec(2, 2, vec![0.0, 1.0, 2.0, 0.0]);

    let nn = NN::<LeakyIntegrateFire>::new()
        .layer(v.clone(), input_weights0, intra_weights0)?
        .layer(v.clone(), input_weights1, intra_weights1);

    return nn;
}
pub fn recursive_funct(input: u8, layer:&Layer<LeakyIntegrateFire>){

    let next_layer=todo!();
    let neurons = &layer.neurons;
    for neuron in neurons {
        print!("neuron: ");
        //do computations
        let new_input = 1; //or 0 
        if (1 > 0) {
            //add check for spike generation
            //genera un numero d spike pari al numero di neurons nel layer dopo
            println!("spike generation");

            recursive_funct(new_input,next_layer); //dovrei passare il layer successivo 
        }
    }
}

pub fn start(nn: NN<LeakyIntegrateFire>, input_vec: Vec<u8>) {
    let layers = nn.layers;

    let L0 = &layers[0];

    let first_input = input_vec[0];

    recursive_funct(first_input, L0)

    
}
fn main() {
    let nn = f2().unwrap();
    let input = vec![0, 1];
    start(nn, input);
}
