pub mod snn;

use std::collections::HashMap;
use std::io;
use std::iter::Map;
use std::str::FromStr;
use crate::snn::model::{Model, Stuck};
use crate::snn::resilience::Resilience;
use nalgebra::{DMatrix, DVector};
use ndarray::Array;
use crate::snn::model::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
use itertools::Itertools;
use crate::snn::nn::NN;
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

fn _matrix_mul(){
    let mat1 = DMatrix::from_vec(3, 3, vec![
        1.0, 2.0, 0.0,
        0.0,1.0,0.0,
        0.0,0.0,1.0]);
    println!("Matrix: {}",mat1);
    // Crea il vettore 2x1 mat2
    let mat2 = DVector::from_vec(vec![1.0, 1.0,1.0]);
    println!("{}",mat2);

    // Esegui la moltiplicazione tra mat1 e mat2
    let result = mat2.transpose()*mat1;


    // Stampa il risultato
    println!("Risultato:\n{}", result);
}

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
/*
fn test_solve(nn:NN<LeakyIntegrateFire>, input: Vec<u128>) {
    let mut nn_temp = nn.clone();
    nn_temp.solve_single_vec_spike(input);
}
*/
fn main() {

    let mut nn = create_neural_network_from_user_input();

    // Input per la verifica della resilienza
    let (components, stuck_type, num_trials) = get_resilience_test_input();

    // Esecuzione della verifica della resilienza
    let neural_network = match nn {
        Ok(nn) => nn,
        Err(e) => {
            println!("Errore nella definizione della rete neurale: {}",e.to_string())
        }
    };
    let success_percentage = execute_resilience_test(&neural_network, components, stuck_type, num_trials);

    println!("Percentuale di successo: {:.2}%", success_percentage);



    //let nn = f2().unwrap();
    //let input = vec![0,2,5];
    //matrix_mul();

    //nn.solve_single_thread(input);
    // let configuration: Resilience = Resilience::new(vec!["Neurons".to_string(), "Input Links".to_string()], Stuck::Zero, 5);
    // println!("Il componente scelto è: {}", configuration.get_rand_component());
    //test_solve(nn,input);

}

fn create_neural_network_from_user_input() -> Result<NN<LeakyIntegrateFire>, String> {
    let mut nn = NN::<LeakyIntegrateFire>::new();

    let mut config_map: HashMap<String, Configuration> = HashMap::new();
    let mut first_layer = true;
    let mut prev_layer_neurons= 0;
    loop {
        loop {
            println!("Inserisci il nome della configurazione (o 'fine' per terminare): ");
            let mut conf_name = String::new();
            io::stdin().read_line(&mut conf_name).expect("Errore nella lettura dell'input");
            if conf_name.trim().eq_ignore_ascii_case("fine") {
                break;
            }

            let mut v_rest = String::new();
            println!("Inserisci il potenziale di riposo (o 'fine' per terminare):");
            io::stdin().read_line(& mut v_rest).expect("Errore nella lettura della v_rest");
            if v_rest.trim().eq_ignore_ascii_case("fine") {
                break;
            }
            let v_rest_f = match v_rest.trim().parse::<f64>() {
                Ok(num) => {println!("Il numero è: {}", num); num},
                Err(e) => {println!("Errore nella conversione: {}", e); return Err(e.to_string()); },
            };
            let mut v_reset = String::new();
            println!("Inserisci il potenziale di reset (o 'fine' per terminare):");
            io::stdin().read_line(& mut v_reset).expect("Errore nella lettura della v_reset");
            if v_reset.trim().eq_ignore_ascii_case("fine") {
                break;
            };
            let v_reset_f = match v_reset.trim().parse::<f64>() {
                Ok(num) => num,
                Err(e) => return Err(e.to_string()),
            };
            let mut v_threshold = String::new();
            println!("Inserisci il potenziale di soglia (o 'fine' per terminare):");
            io::stdin().read_line(& mut v_threshold).expect("Errore nella lettura della v_threshold");

            if v_threshold.trim().eq_ignore_ascii_case("fine") {
                break;
            }
            let v_threshold_f = match v_threshold.trim().parse::<f64>() {
                Ok(num) => {println!("Il numero è: {}", num); num},
                Err(e) => {println!("Errore nella conversione: {}", e); return Err(e.to_string()); },
            };
            let mut tau = String::new();
            println!("Inserisci tau (o 'fine' per terminare):");
            io::stdin().read_line(& mut tau).expect("Errore nella lettura della tau");

            if tau.trim().eq_ignore_ascii_case("fine") {
                break;
            };
            let tau_f = match tau.trim().parse::<f64>() {
                Ok(num) => {println!("Il numero è: {}", num); num},
                Err(e) => {println!("Errore nella conversione: {}", e); return Err(e.to_string()); },
            };
            println!("conf name: {}", conf_name.clone());
            config_map.insert(conf_name, Configuration::new(v_rest_f,v_reset_f, v_threshold_f, tau_f));
        } // end loop interno

        if(first_layer){
            first_layer = false;
            println!("Inserisci il vettore delle configurazioni dei neuroni del primo layer: (o fine per terminare)");
            let mut input_neurons = String::new();
            io::stdin().read_line(&mut input_neurons).expect("Errore nella lettura dell'input");
            if input_neurons.trim().eq_ignore_ascii_case("fine") {
                break;
            }
            let input_neurons_len = input_neurons.split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len();
            println!("Inserisci i pesi del primo layer (o fine per terminare): ");
            let mut input_weights = String::new();
            io::stdin().read_line(&mut input_weights).expect("Errore nella lettura degli input weights");
            println!("pesi primo layer: {:?} \n Lunghezza primo layer: {}",input_weights.split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>(),input_weights.split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len());
            println!("neuroni primo layer: {:?} \n Lunghezza neuroni primo layer: {}",input_neurons.split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>(),input_neurons_len);
            if input_weights.split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len() != input_neurons_len {
                return Err("Numero di input weights sbagliato".to_string());
            }
            let mut row = 0;
            let mut intra_weights: Vec<String> = Vec:: new();
            loop {
                intra_weights.push("".to_string());
                println!("Inserisci i pesi interni della riga {} del primo layer (o fine per terminare): ", row);
                io::stdin().read_line(&mut intra_weights[row]).expect("Errore nella lettura degli intra weights");
                if intra_weights[row].split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len() != input_neurons_len {
                    return Err("Numero di intra weights sbagliato".to_string());
                }
                row+=1;
                if row == input_neurons.split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len() {
                    break;
                }
            } // end loop
            // creo la matrice diagonale di input weights
            prev_layer_neurons = input_neurons_len;
            let diagonal_values: Vec<f64> = input_weights
                .split_whitespace()
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .collect();
            let n = diagonal_values.len();
            let mut matrix = DMatrix::from_diagonal_element(n, n, 0.0); // Crea una matrice n x n con zero
            for (i, value) in diagonal_values.iter().enumerate() {
                matrix[(i, i)] = *value;
            }
            let neurons: Result<Vec<LifNeuron>, String> = input_neurons
                .split_whitespace()
                .map(|config| {
                    config_map.get(config)
                        .ok_or_else(|| format!("Configurazione non trovata per la chiave: {}", config))
                        .map(|conf| LifNeuron::from_conf(conf))
                })
                .collect();
            nn= nn.layer(
                match neurons {
                    Ok(neurons) => {
                        neurons
                    },
                    Err(e) => {
                        return Err(e.to_string())
                    }
                },
                matrix,
                DMatrix::from_vec(
                    input_neurons_len,
                    input_neurons_len,
                    intra_weights
                        .iter().join(" ")
                        .split_whitespace()
                        .filter_map(|s| s.trim().parse::<f64>().ok())
                        .collect()
                )
            ).expect("Errore new layer");
        } // end first layer
        let mut layer_idx = 1;
        loop {
            first_layer = false;
            println!("Inserisci il vettore delle configurazioni dei neuroni del layer {}: (o fine per terminare)", layer_idx);
            let mut input_neurons = String::new();
            io::stdin().read_line(&mut input_neurons).expect("Errore nella lettura dell'input");
            if input_neurons.trim().eq_ignore_ascii_case("fine") {
                break;
            }
            let input_neurons_len = input_neurons.split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len();
            let mut row = 0;
            let mut input_weights: Vec<String> = Vec::new();

            loop {
                println!("Inserisci i pesi del layer {}, riga {} (o fine per terminare): ", layer_idx, row);
                io::stdin().read_line(&mut input_weights[row]).expect("Errore nella lettura degli input weights");
                if input_neurons.trim().eq_ignore_ascii_case("fine") {
                    break;
                }
                if input_weights[row].split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len() != input_neurons_len {
                    return Err("Numero di input weights sbagliato".to_string());
                }
                if row == prev_layer_neurons {
                    break;
                }
            } // end loop input weights
            row = 0;
            let mut intra_weights: Vec<String> = Vec::new();

            loop {
                println!("Inserisci i pesi interni del layer {}, riga {} del primo layer (o fine per terminare): ",layer_idx, row);
                io::stdin().read_line(&mut intra_weights[row]).expect("Errore nella lettura degli intra weights");

                if intra_weights[row].split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len() != input_neurons.split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len() {
                    return Err("Numero di intra weights sbagliato".to_string());
                }
                row+=1;
                if row == input_neurons.split_whitespace().map(|s| s.to_string()).collect::<Vec<String>>().len() {
                    break;
                }
            }
            layer_idx+=1;
            prev_layer_neurons = input_neurons.len();
            nn = nn.layer(
                input_neurons.split(" ").map(|config| LifNeuron::from_conf(&config_map[config])).collect(),
                DMatrix::from_vec(
                    prev_layer_neurons,
                    input_neurons_len,
                    input_weights
                        .iter().join(" ")
                        .split_whitespace()
                        .filter_map(|s| s.trim().parse::<f64>().ok())
                        .collect()
                ),
                DMatrix::from_vec(
                    input_neurons_len,
                    input_neurons_len,
                    intra_weights
                        .iter().join(" ")
                        .split_whitespace()
                        .filter_map(|s| s.trim().parse::<f64>().ok())
                        .collect()
                ),
            ).expect("Errore");
        } // end loop other layers
    }

    return Ok(nn);
}


fn get_resilience_test_input() -> (Vec<String>, Stuck, usize) {
    let mut input_components = String::new();
    println!("Inserisci la lista dei componenti su cui iniettare il bit: ");
    io::stdin().read_line(&mut input_components).expect("Errore nella lettura dei componenti");
    let mut stuck= String::new();
    println!("Inserisci il tipo di stuck (zero, one, transient): ");
    io::stdin().read_line(&mut stuck).expect("Errore nella lettura del tipo di stuck");
    let mut num= String::new();
    println!("Inserisci il numero di test da fare: ");
    io::stdin().read_line(&mut num).expect("Errore nella lettura del numero di test da fare");

    return (
        vec![input_components.split_whitespace().collect()],
        match &stuck as &str {
            "zero" => Stuck::Zero,
            "one" => Stuck::One,
            "transient" => Stuck::Transient,
            _ => {
                Stuck::Zero
            }
        },
        num.trim().parse::<usize>().ok().unwrap()
    )
}

fn execute_resilience_test(nn: &NN<LeakyIntegrateFire>, components: Vec<String>, stuck_type: Stuck, num_trials: usize) -> f64 {
    let configuration: Resilience = Resilience::new(components, stuck_type, num_trials as u128);
    let spikes=vec![
        (0, vec![1,2,5,7,8,10,11]),
        (1, vec![1,2,5,7,8,10,11]),
        (2, vec![1,2,5,7,8,10,11]),
    ];
    configuration.execute_resilience_test(nn.clone(),spikes);
    return 0.0;
}
