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
    let mut nn = create_neural_network_from_user_input();

    match nn {
        Ok(nn) => {
            // Do something with the neural network if needed
            println!("Neural network created successfully!");
        }
        Err(err) => {
            // Handle the error and print a message
            eprintln!("Error creating neural network: {}", err);
        }
    }

    /*
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


    */
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

        let result = read_layer_configurations(&config_map, layer_idx, prev_layer_len)?;
        prev_layer_len = result.0.len(); //the vector of neurons
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
        0
    )?;
    let intra_weights = read_matrix_input(
        "Inserisci i pesi interni del layer : ",
        input_neurons_len,
        layer_idx,
        prev_layer_len,
        1
    )?;

    Ok((neurons, input_weights, intra_weights))
}

fn read_matrix_input(
    prompt: &str,
    size: usize,
    layer_idx: usize,
    prev_layer_len: usize,
    selector:usize
) -> Result<DMatrix<f64>, String> {
    let mut matrix_values = Vec::new();
    let max_row = if selector == 0 { prev_layer_len } else { size };
    
    for row in 0..max_row  {
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
            break;
        }
        let trimmed_conf_name = conf_name.trim().to_string();

        let config = create_configuration()?;
        config_map.insert(trimmed_conf_name, config);
    }

    Ok(())
}

fn get_resilience_test_input() -> (Vec<String>, Stuck, usize) {
    let mut input_components = String::new();
    println!("Inserisci la lista dei componenti su cui iniettare il bit: ");
    io::stdin()
        .read_line(&mut input_components)
        .expect("Errore nella lettura dei componenti");
    let mut stuck = String::new();
    println!("Inserisci il tipo di stuck (zero, one, transient): ");
    io::stdin()
        .read_line(&mut stuck)
        .expect("Errore nella lettura del tipo di stuck");
    let mut num = String::new();
    println!("Inserisci il numero di test da fare: ");
    io::stdin()
        .read_line(&mut num)
        .expect("Errore nella lettura del numero di test da fare");

    return (
        vec![input_components.split_whitespace().collect()],
        match stuck.to_lowercase().trim()  {
            "zero" => Stuck::Zero,
            "one" => Stuck::One,
            "transient" => Stuck::Transient,
            _ => Stuck::Zero,
        },
        num.trim().parse::<usize>().ok().unwrap(),
    );
}

fn execute_resilience_test(
    nn: &NN<LeakyIntegrateFire>,
    components: Vec<String>,
    stuck_type: Stuck,
    num_trials: usize,
) -> f64 {
    let configuration: Resilience = Resilience::new(components, stuck_type, num_trials as u128);
    let spikes = vec![
        (0, vec![1, 2, 5, 7, 8, 10, 11]),
        (1, vec![1, 2, 5, 7, 8, 10, 11]),
        (2, vec![1, 2, 5, 7, 8, 10, 11]),
    ];
    configuration.execute_resilience_test(nn.clone(), spikes);
    return 0.0;
}

/* fn old() -> Result<NN<LeakyIntegrateFire>, String> {
    let mut nn = NN::<LeakyIntegrateFire>::new();

    let mut config_map: HashMap<String, Configuration> = HashMap::new();
    let mut first_layer = true;
    let mut prev_layer_len = 0;

    //define configuration loop
    loop {
        println!("Inserisci il nome della configurazione (o 'fine' per terminare): ");
        let mut conf_name = String::new();
        io::stdin()
            .read_line(&mut conf_name)
            .expect("Errore nella lettura dell'input");
        if conf_name.trim().eq_ignore_ascii_case("fine") {
            break;
        }
        let trimmed_conf_name = conf_name.trim().to_string();

        let mut v_rest = String::new();
        println!("Inserisci il potenziale di riposo (o 'fine' per terminare):");
        io::stdin()
            .read_line(&mut v_rest)
            .expect("Errore nella lettura della v_rest");
        if v_rest.trim().eq_ignore_ascii_case("fine") {
            break;
        }
        let v_rest_f = match v_rest.trim().parse::<f64>() {
            Ok(num) => {
                println!("Il numero è: {}", num);
                num
            }
            Err(e) => {
                println!("Errore nella conversione: {}", e);
                return Err(e.to_string());
            }
        };
        let mut v_reset = String::new();
        println!("Inserisci il potenziale di reset (o 'fine' per terminare):");
        io::stdin()
            .read_line(&mut v_reset)
            .expect("Errore nella lettura della v_reset");
        if v_reset.trim().eq_ignore_ascii_case("fine") {
            break;
        };
        let v_reset_f = match v_reset.trim().parse::<f64>() {
            Ok(num) => num,
            Err(e) => return Err(e.to_string()),
        };
        let mut v_threshold = String::new();
        println!("Inserisci il potenziale di soglia (o 'fine' per terminare):");
        io::stdin()
            .read_line(&mut v_threshold)
            .expect("Errore nella lettura della v_threshold");

        if v_threshold.trim().eq_ignore_ascii_case("fine") {
            break;
        }
        let v_threshold_f = match v_threshold.trim().parse::<f64>() {
            Ok(num) => {
                println!("Il numero è: {}", num);
                num
            }
            Err(e) => {
                println!("Errore nella conversione: {}", e);
                return Err(e.to_string());
            }
        };

        let mut tau = String::new();
        println!("Inserisci tau (o 'fine' per terminare):");
        io::stdin()
            .read_line(&mut tau)
            .expect("Errore nella lettura della tau");

        if tau.trim().eq_ignore_ascii_case("fine") {
            break;
        };
        let tau_f = match tau.trim().parse::<f64>() {
            Ok(num) => {
                println!("Il numero è: {}", num);
                num
            }
            Err(e) => {
                println!("Errore nella conversione: {}", e);
                return Err(e.to_string());
            }
        };
        println!("conf name: {}", trimmed_conf_name.clone());
        config_map.insert(
            trimmed_conf_name,
            Configuration::new(v_rest_f, v_reset_f, v_threshold_f, tau_f),
        );
        println!("map {:?}", config_map);
    } // end loop

    //first layer creation
    if (first_layer) {
        first_layer = false;
        println!("Inserisci il vettore delle configurazioni dei neuroni del primo layer: (o fine per terminare)");
        //inserisce c1 c2 c1 ....
        let mut input_neurons = String::new();
        io::stdin()
            .read_line(&mut input_neurons)
            .expect("Errore nella lettura dell'input");
        if input_neurons.trim().eq_ignore_ascii_case("fine") {
            break;
        }
        let input_neurons_len = input_neurons
            .split_whitespace()
            .map(|s| s.to_string())
            .collect::<Vec<String>>()
            .len();

        println!("Inserisci i pesi del primo layer (o fine per terminare): ");
        //inizialmente è una matrice diagonale con diagonale lunga quano il numero di neuroni del primo layer
        let mut input_weights = String::new();
        io::stdin()
            .read_line(&mut input_weights)
            .expect("Errore nella lettura degli input weights");

        let vec_input_weights = input_weights
            .split_whitespace()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        println!(
            "pesi primo layer: {:?} \n Lunghezza primo layer: {}",
            vec_input_weights,
            vec_input_weights.len()
        );
        println!(
            "neuroni primo layer: {:?} \n Lunghezza neuroni primo layer: {}",
            input_neurons
                .split_whitespace()
                .map(|s| s.to_string())
                .collect::<Vec<String>>(),
            input_neurons_len
        );
        if vec_input_weights.len() != input_neurons_len {
            return Err("Numero di input weights sbagliato".to_string());
        }

        // creo la matrice diagonale di input weights
        prev_layer_len = input_neurons_len;
        let diagonal_values: Vec<f64> = input_weights
            .split_whitespace()
            .filter_map(|s| s.trim().parse::<f64>().ok())
            .collect();
        let n = diagonal_values.len();
        println!("diagonal values: {:?}  with len {}", diagonal_values, n);
        let mut matrix = DMatrix::from_diagonal_element(n, n, 0.0); // Crea una matrice n x n con zero
        matrix.set_diagonal(&Vector::from(diagonal_values));
        println!("Resulting input matrix:\n{}", matrix);

        //creao intra matrix
        let mut row = 0;
        let mut intra_weights: Vec<String> = Vec::new();
        loop {
            intra_weights.push("".to_string());
            println!(
                "Inserisci i pesi interni della riga {} del primo layer (o fine per terminare): ",
                row
            );
            io::stdin()
                .read_line(&mut intra_weights[row])
                .expect("Errore nella lettura degli intra weights");
            if intra_weights[row]
                .split_whitespace()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .len()
                != input_neurons_len
            {
                return Err("Numero di intra weights sbagliato".to_string());
            }
            row += 1;
            if row
                == input_neurons
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>()
                    .len()
            {
                break;
            }
        } // end loop

        // Create the LifNeuron instances from configurations stored in input_neurons
        let neurons: Vec<LifNeuron> = input_neurons
            .split_whitespace()
            .map(|config_key| {
                let configuration = config_map.get(config_key).expect(&format!(
                    "Configurazione non trovata per la chiave: {}",
                    config_key
                ));
                LifNeuron::from_conf(configuration)
            })
            .collect();
        println!("neurons: {:?}", neurons);

        println!("creating first layer nn");
        //finally create the nn
        nn = nn
            .layer(
                neurons,
                matrix,
                DMatrix::from_vec(
                    input_neurons_len,
                    input_neurons_len,
                    intra_weights
                        .iter()
                        .join(" ")
                        .split_whitespace()
                        .filter_map(|s| s.trim().parse::<f64>().ok())
                        .collect(),
                ),
            )
            .expect("Errore new layer");
    }
    // end first layer

    let mut layer_idx = 1;
    loop {
        first_layer = false;
        println!("Inserisci il vettore delle configurazioni dei neuroni del layer {}: (o fine per terminare)", layer_idx);
        let mut input_neurons = String::new();
        io::stdin()
            .read_line(&mut input_neurons)
            .expect("Errore nella lettura dell'input");
        if input_neurons.trim().eq_ignore_ascii_case("fine") {
            break;
        }
        let input_neurons_len = input_neurons
            .split_whitespace()
            .map(|s| s.to_string())
            .collect::<Vec<String>>()
            .len();

        let mut row = 0;
        let mut input_weights: Vec<String> = Vec::with_capacity(prev_layer_len);
        //create input weights loop
        loop {
            println!("Inserisci i pesi del layer {}, riga {} : ", layer_idx, row);
            let mut input_string = String::new();

            io::stdin()
                .read_line(&mut input_string)
                .expect("Errore nella lettura degli input weights");

            input_weights.push(input_string.trim().to_string());

            if input_weights[row]
                .split_whitespace()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .len()
                != input_neurons_len
            {
                return Err("Numero di input weights sbagliato".to_string());
            }
            row += 1;
            if row == prev_layer_len {
                break;
            }
        } // end loop input weights

        row = 0;
        let mut intra_weights: Vec<String> = Vec::new();
        //create intra weights loop
        loop {
            println!(
                "Inserisci i pesi interni del layer {}, riga {} : ",
                layer_idx, row
            );
            let mut intra_string = String::new();
            io::stdin()
                .read_line(&mut intra_string)
                .expect("Errore nella lettura degli intra weights");
            intra_weights.push(intra_string.trim().to_string());

            if intra_weights[row]
                .split_whitespace()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .len()
                != input_neurons
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>()
                    .len()
            {
                return Err("Numero di intra weights sbagliato".to_string());
            }
            row += 1;
            if row
                == input_neurons
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>()
                    .len()
            {
                break;
            }
        }
        //end loop

        // Create the LifNeuron instances from configurations stored in input_neurons
        let neurons: Vec<LifNeuron> = input_neurons
            .split_whitespace()
            .map(|config_key| {
                println!("{}", config_key);
                let configuration = config_map.get(config_key).expect(&format!(
                    "Configurazione non trovata per la chiave: {}",
                    config_key
                ));
                LifNeuron::from_conf(configuration)
            })
            .collect();
        println!("neurons: {:?}", neurons);
        println!("updating nn with the new layer_idx: {}", layer_idx);
        nn = nn
            .layer(
                neurons,
                DMatrix::from_vec(
                    prev_layer_len,
                    input_neurons_len,
                    input_weights
                        .iter()
                        .join(" ")
                        .split_whitespace()
                        .filter_map(|s| s.trim().parse::<f64>().ok())
                        .collect(),
                ),
                DMatrix::from_vec(
                    input_neurons_len,
                    input_neurons_len,
                    intra_weights
                        .iter()
                        .join(" ")
                        .split_whitespace()
                        .filter_map(|s| s.trim().parse::<f64>().ok())
                        .collect(),
                ),
            )
            .expect("Errore");

        layer_idx += 1;
        prev_layer_len = input_neurons
            .split_whitespace()
            .map(|s| s.to_string())
            .collect::<Vec<String>>()
            .len();
    } // end loop other layers

    return Ok(nn);
}
 */
