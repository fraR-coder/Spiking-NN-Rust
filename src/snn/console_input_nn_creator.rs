use std::collections::HashMap;
use std::io;

use crate::snn::model::lif::*;
use crate::NN;


use crate::snn::model::Stuck;
use nalgebra::DMatrix;

/// Creates a neural network from user input.
///
/// This function prompts the user to input configurations for each layer of a neural network.
/// It populates the layers with neurons, input weights, and intra-layer weights.
///
/// # Returns
///
/// A `Result` containing the neural network (`NN<LeakyIntegrateFire>`) on success,
/// or an error message (`String`) on failure.
///
/// # Examples
///


pub fn create_neural_network_from_user_input() -> Result<NN<LeakyIntegrateFire>, String> {
    let mut nn = NN::<LeakyIntegrateFire>::new();
    let mut config_map: HashMap<String, Configuration> = HashMap::new();

    //populate the hashmap
    read_neuron_configurations(&mut config_map)?;

    let mut prev_layer_len = 0;
    let mut layer_idx = 0;

    loop {
        println!("Premi qualsiasi tasto per inserire un nuovo layer? digita 'fine' altrimenti");
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

/// Reads configurations for a neural network layer from the user.
///
/// This function is used internally by `create_neural_network_from_user_input`.

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
        return Err("The number of neurons should be at least 1".to_string());
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

/// Reads matrix input from the user for weights configuration.
///
/// # Parameters
///
/// - `prompt`: A prompt message for the user.
/// - `size`: The size of the matrix.
/// - `layer_idx`: The index of the layer.
/// - `prev_layer_len`: The size of the previous layer.
/// - `selector`: A selector indicating whether it's input or intra weights.
///
/// # Returns
///
/// A `Result` containing the matrix (`DMatrix<f64>`) on success,
/// or an error message (`String`) on failure.
///

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
    //println!("Matrix values length: {}", matrix_values.len());
    let matrix = DMatrix::from_vec(max_row, size, matrix_values);
    println!("matrice {}", matrix);
    Ok(matrix)
}

/// Reads a floating-point number from the user.
///
/// This function is used internally for configuration input.
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
            println!("Inserisci un numero valido.");
        }
    }
}

/// Creates a neuron configuration based on user input.
///
/// # Returns
///
/// A `Result` containing the neuron configuration (`Configuration`) on success,
/// or an error message (`String`) on failure.
fn create_configuration() -> Result<Configuration, String> {
    let v_rest = read_f64_input("Inserisci il potenziale di riposo :");
    let v_reset = read_f64_input("Inserisci il potenziale di reset :");
    let v_threshold = read_f64_input("Inserisci il potenziale di soglia :");
    let tau = read_f64_input("Inserisci tau :");

    Ok(Configuration::new(v_rest, v_reset, v_threshold, tau))
}

/// Reads neuron configurations from the user and populates a configuration map.
///
/// This function is used during the creation of the neural network.
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
            if config_map.keys().count() == 0 {
                println!("inserisci almeno 1 configurazione");
                read_neuron_configurations(config_map).expect("Error read neuron configurations");
            }
            break;
        }
        let trimmed_conf_name = conf_name.trim().to_string();

        let config = create_configuration()?;
        config_map.insert(trimmed_conf_name, config);
    }

    Ok(())
}

/// Gets input for resilience testing.
///
/// # Returns
///
/// A tuple containing the list of components, stuck type, and the number of trials.
///
pub fn get_resilience_test_input() -> (Vec<String>, Stuck, usize) {
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

/// Reads spike vectors from the user based on the number of neurons.
///
/// # Parameters
///
/// - `num_neurons`: The number of neurons for which to read spike vectors.
///
/// # Returns
///
/// A vector of spikes, where each spike is a tuple containing a neuron index and a vector of instants.
///
pub fn read_spike_vector(num_neurons: usize) -> Vec<(u128, Vec<u128>)> {
    let mut spikes = Vec::new();

    for i in 0..num_neurons {
        let mut input = String::new();
        println!("Inserisci il vettore di spike per il neurone {} (usa lo spazio come separatore): ", i);
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

#[derive(PartialEq)]
pub enum CreationMode{
    FromFile,
    FromTerminal
}
pub fn read_snn_creation_mode() -> CreationMode{
    let mut input = String::new();
    println!("Come vuoi inserire la rete neurale? \n- 1 per leggere la rete dai file di configurazione\n- 2 per inserire la rete da terminale");
    io::stdin()
        .read_line(&mut input)
        .expect("Errore nella lettura dell'input");
    if input.trim().parse::<i32>().ok().unwrap()==1{
        return CreationMode::FromFile
    } else {
        return CreationMode::FromTerminal
    }

}
