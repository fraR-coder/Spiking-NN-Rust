pub mod snn;

use crate::snn::model::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
use crate::snn::model::{Model, Stuck};
use crate::snn::nn::NN;
use crate::snn::resilience::Resilience;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector, Vector};
use ndarray::Array;
use std::collections::HashMap;
use std::io;
use std::iter::Map;
use std::str::FromStr;
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

fn _matrix_mul() {
    let mat1 = DMatrix::from_vec(3, 3, vec![1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    println!("Matrix: {}", mat1);
    // Crea il vettore 2x1 mat2
    let mat2 = DVector::from_vec(vec![1.0, 1.0, 1.0]);
    println!("{}", mat2);

    // Esegui la moltiplicazione tra mat1 e mat2
    let result = mat2.transpose() * mat1;

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
    let nn = create_neural_network_from_user_input();

    match nn {
        Ok(nn) => {
            println!("Neural network created successfully!");
            if nn.get_num_layers() == 0 {
                eprintln!("Error: the neural network is empty");
                return;
            }
            let (components, stuck_type, num_trials) = get_resilience_test_input();
            let configuration: Resilience =
                Resilience::new(components, stuck_type, num_trials as u128);
            let first_neurons=nn.layers.get(0).unwrap().num_neurons();
            let spikes=read_spike_vector(first_neurons) ; 
            println!("spikes: {:?}", spikes);
            configuration.execute_resilience_test(nn.clone(), spikes);
        }
        Err(err) => {
            // Handle the error and print a message
            eprintln!("Error creating neural network: {}", err);
        }
    }
}

fn create_neural_network_from_user_input() -> Result<NN<LeakyIntegrateFire>, String> {
    let mut nn = NN::<LeakyIntegrateFire>::new();
    let mut config_map: HashMap<String, Configuration> = HashMap::new();

    //populate the hashmap
    read_neuron_configurations(&mut config_map)?;

    let mut prev_layer_len = 0;
    let mut layer_idx = 0;

    loop {
        println!("Do you want to insert a new layer? Type 'fine' to stop.");
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Error reading input");
        if input.trim().eq_ignore_ascii_case("fine") {
            break;
        }

        let (neurons, input_weights, intra_weights) =
            read_layer_configurations(&config_map, layer_idx, prev_layer_len)?;

        nn = nn
            .layer(neurons.clone(), input_weights, intra_weights)
            .expect(format!("Error adding layer {}", layer_idx).as_str());

        prev_layer_len = neurons.len(); //the vector of neurons
        layer_idx += 1;
    }

    Ok(nn)
}

fn read_layer_configurations(
    config_map: &HashMap<String, Configuration>,
    layer_idx: usize,
    mut prev_layer_len: usize,
) -> Result<(Vec<LifNeuron>, DMatrix<f64>, DMatrix<f64>), String> {
    println!("Inserisci il vettore delle configurazioni dei neuroni : ");
    let mut input_neurons = String::new(); //something like c1 c2 c3 c1 ...
    io::stdin()
        .read_line(&mut input_neurons)
        .expect("Errore nella lettura dell'input");

    if input_neurons.trim().is_empty() {
        //Err("The number of neurons should be at least 1".to_string());

    }
    let input_neurons_len = input_neurons.split_whitespace().count();
    //if zero is the first layer and the number of inputs is equal to number of neurons
    if prev_layer_len == 0 {
        prev_layer_len = input_neurons_len;
    }
    //obtain the Lifneuron from the ConfigurationMap
    let neurons: Vec<LifNeuron> = input_neurons
        .split_whitespace()
        .map(|config_key| {
            config_map
                .get(config_key)
                .ok_or_else(|| format!("Configurazione non trovata per la chiave: {}", config_key))
                .and_then(|conf| Ok(LifNeuron::from_conf(conf)))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let input_weights = read_matrix_input(
        "Inserisci i pesi in input al layer : ",
        input_neurons_len,
        layer_idx,
        prev_layer_len,
        0,
    )?;
    let intra_weights = read_matrix_input(
        "Inserisci i pesi interni del layer : ",
        input_neurons_len,
        layer_idx,
        prev_layer_len,
        1,
    )?;

    Ok((neurons, input_weights, intra_weights))
}

fn read_matrix_input(
    prompt: &str,
    size: usize,
    layer_idx: usize,
    prev_layer_len: usize,
    selector: usize,
) -> Result<DMatrix<f64>, String> {
    let mut matrix_values = Vec::new();
    let max_row = if selector == 0 { prev_layer_len } else { size };

    for row in 0..max_row {
        println!("{}{} riga {}:", prompt, layer_idx, row);
        let mut row_input = String::new();
        io::stdin()
            .read_line(&mut row_input)
            .expect("Errore nella lettura degli input");

        let values: Vec<f64> = row_input
            .split_whitespace()
            .map(|s| s.parse().ok())
            .flatten()
            .collect();

        if values.len() != size {
            return Err(format!("Numero di pesi sbagliato nella riga {}", row));
        }

        matrix_values.extend(values);
    }
    println!("Matrix values length: {}", matrix_values.len());
    let matrix = DMatrix::from_vec(max_row, size, matrix_values);
    println!("matrix {}", matrix);
    Ok(matrix)
}

fn read_f64_input(prompt: &str) -> f64 {
    loop {
        println!("{}", prompt);
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Error reading input");

        if let Ok(value) = input.trim().parse() {
            return value;
        } else {
            println!("Invalid input. Please enter a valid number.");
        }
    }
}

fn create_configuration() -> Result<Configuration, String> {
    let v_rest = read_f64_input("Inserisci il potenziale di riposo :");
    let v_reset = read_f64_input("Inserisci il potenziale di reset :");
    let v_threshold = read_f64_input("Inserisci il potenziale di soglia :");
    let tau = read_f64_input("Inserisci tau :");

    Ok(Configuration::new(v_rest, v_reset, v_threshold, tau))
}

fn read_neuron_configurations(
    config_map: &mut HashMap<String, Configuration>,
) -> Result<(), String> {
    loop {
        println!("Inserisci il nome della configurazione (o 'fine' per terminare): ");
        let mut conf_name = String::new();
        io::stdin()
            .read_line(&mut conf_name)
            .expect("Errore nella lettura dell'input");
        if conf_name.trim().eq_ignore_ascii_case("fine") {
            if config_map.keys().count()==0 {
                println!("inserisci almeno 1 configurazione");
                read_neuron_configurations(config_map);
            }
            break;
        }
        let trimmed_conf_name = conf_name.trim().to_string();

        let config = create_configuration()?;
        config_map.insert(trimmed_conf_name, config);
    }

    Ok(())
}

fn get_resilience_test_input() -> (Vec<String>, Stuck, usize) {
    fn read_input(prompt: &str) -> Result<Vec<String>, io::Error> {
        let mut input = String::new();
        println!("{}", prompt);
        io::stdin().read_line(&mut input)?;

        Ok(input
            .trim()
            .split(',')
            .map(|s| s.trim().to_string())
            .collect())
    }

    let input_components = read_input(
        "Inserisci la lista dei componenti su cui iniettare il bit separati da virglole: ",
    )
    .expect("Error reading components");

    let stuck = if let Some(stuck_type) =
        read_input("Inserisci il tipo di stuck (zero, one, transient): ")
            .expect("Error reading stuck type")
            .get(0)
    {
        match stuck_type.to_lowercase().as_str() {
            "zero" => Stuck::Zero,
            "one" => Stuck::One,
            "transient" => Stuck::Transient,
            _ => Stuck::Zero,
        }
    } else {
        Stuck::Zero // Default value if the input vector is empty
    };

    let num = read_input("Inserisci il numero di test da fare: ")
        .and_then(|s| {
            s[0].parse::<usize>()
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        })
        .expect("Error reading number of tests");

    (input_components, stuck, num)
}

fn read_spike_vector(num_neurons: usize) -> Vec<(u128,Vec<u128>)> {
    let mut spikes = Vec::new();

    for i in 0..num_neurons {
        let mut input = String::new();
        println!("Inserisci il vettore di spike per il neurone {}: ", i);
        io::stdin()
            .read_line(&mut input)
            .expect("Errore nella lettura dell'input");

        let instants: Vec<u128> = input
            .trim()
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();

        let spike = (i as u128, instants);
        spikes.push(spike);
    }

    spikes
}
