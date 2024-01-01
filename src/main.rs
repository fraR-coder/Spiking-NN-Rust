pub mod snn;

use crate::snn::console_input_nn_creator::*;
use crate::snn::model::Model;
use crate::snn::resilience::Resilience;
use snn::console_input_nn_creator::create_neural_network_from_user_input;
use snn::json_adapter::{InputJson, NeuronJson};
use crate::snn::json_adapter::{PathsJson, ResilienceJson};

fn main() {
    let creation_mode = read_snn_creation_mode();
    if creation_mode==CreationMode::FromFile{
        let paths: PathsJson = match PathsJson::read_from_file("src/configuration/paths.json") {
            Ok(paths_json) => Some(paths_json), // In caso di successo, incapsula i paths in Some
            Err(error_message) => {
                eprintln!("Errore durante la lettura del file dei path: {}", error_message);
                if error_message.contains("Si prega di compilare con i path appropriati.") {
                    None
                } else {
                    std::process::exit(1);
                }
            }
        }.unwrap();

        let nn =
            NeuronJson::read_from_file(&paths.layers,&paths.weights, &paths.configurations);

        let input = InputJson::read_input_from_file(&paths.input_spikes);


        // let configuration: Resilience = Resilience::new(vec!["Neurons".to_string()], Stuck::One, 1000);
        let configuration: Result<Resilience,String> = ResilienceJson::read_from_file(&paths.resilience).expect("Errore lettura file").to_resilience();
        configuration.ok().unwrap().execute_resilience_test(nn.clone().unwrap(),input);
        return;
    } else {
        let nn = create_neural_network_from_user_input();
        match nn {
            Ok(nn) => {
                println!("Spiking Neural Network created successfully");
                if nn.get_num_layers() == 0 {
                    eprintln!("Error: the neural network is empty");
                    return;
                }
                let (components, stuck_type, num_trials) = get_resilience_test_input();
                let configuration: Resilience =
                    Resilience::new(components, stuck_type, num_trials as u128);
                let first_neurons = nn.layers.get(0).unwrap().num_neurons();
                let spikes = read_spike_vector(first_neurons);
                //println!("spikes: {:?}", spikes);
                configuration.execute_resilience_test(nn.clone(), spikes);
            }
            Err(err) => {
                // Handle the error and print a message
                eprintln!("Error creating neural network: {}", err);
            }
        }
    }
}
// use std::str::FromStr;
/*
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

    let s = Spike::new(1, 1,1);

    let s1 = Spike::vec_of_spike_for(1, vec![1, 3, 7]);
    let s2 = Spike::vec_of_spike_for(2, vec![2, 3, 5]);

    let mut vs = Vec::new();
    vs.push(s1);
    vs.push(s2);

    let st = Spike::vec_of_all_spikes(vs);

    println!("{:?}", st);


}
*/

// fn _matrix_mul() {
//     let mat1 = DMatrix::from_vec(3, 3, vec![1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
//     println!("Matrix: {}", mat1);
//     // Crea il vettore 2x1 mat2
//     let mat2 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
//     println!("{}", mat2);
//
//     // Esegui la moltiplicazione tra mat1 e mat2
//     let result = mat2.transpose() * mat1;
//
//     // Stampa il risultato
//     println!("Risultato:\n{}", result);
// }

/*
fn f2() -> Result<NN<LeakyIntegrateFire>, String> {
    let config1 = Configuration::new(2.0, 0.5, 2.0, 1.0);
    let config2 = Configuration::new(2.5, 1.0, 4.0, 1.0);

    let neuron1 = LifNeuron::from_conf(&config1);
    let neuron2 = LifNeuron::from_conf(&config2);

    let mut v = Vec::new();
    v.push(neuron1);
    v.push(neuron2);

    let input_weights0 = DMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
    let intra_weights0 = DMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);

    let input_weights1 = DMatrix::from_vec(2, 2, vec![1.0, 1.0, 2.0, 3.0]);
    let intra_weights1 = DMatrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);

    let nn = NN::<LeakyIntegrateFire>::new()
        .layer(v.clone(), input_weights0, intra_weights0)?
        .layer(v.clone(), input_weights1, intra_weights1);

    return nn;
}

 */


