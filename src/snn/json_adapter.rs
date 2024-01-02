//! # JSON Structs
//!
//! This module defines several structs representing data structures commonly used when
//! reading from or writing to JSON files.
use std::fs::File;
use std::{fs};
use std::io::{ErrorKind, Write};
use crate::snn::model::lif::*;
use crate::NN;

use serde::Deserialize;

use crate::snn::model::Stuck;
use crate::snn::resilience::Resilience;
use nalgebra::DMatrix;

/// Represents the configuration of a neuron read from a JSON file.
#[derive(Debug, Deserialize)]
pub struct NeuronJson {
    neurons: String,
    layers: String,
    configuration: u32,
}
/// Represents a boxed neuron within a neural network layer.

#[derive(Debug)]
pub struct NeuronBox {
    layer: u32,
    position: u32,
    neuron: LifNeuron,
}
/// Represents the weights of a layer in a neural network read from a JSON file.

#[derive(Debug, Deserialize)]
pub struct WeightJson {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}
/// Represents the weights of a layer in a neural network, including input and intra-layer weights.

#[derive(Debug, Deserialize)]
pub struct LayerWeightsJson {
    //layer: u32,
    input_weights: WeightJson,
    intra_weights: WeightJson,
}

#[derive(Debug, Deserialize)]
pub struct InputJson {
    neuron: u128,
    spikes: Vec<u128>,
}
/// Represents the configuration of a neuron in a neural network read from a JSON file.

#[derive(Debug, Deserialize)]
pub struct ConfigurationJson {
    configuration: u32,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64,
}

impl ConfigurationJson {
    /// Reads a vector of neuron configurations from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `pathname` - A string slice representing the path to the JSON file.
    ///
    /// # Returns
    ///
    /// A vector containing neuron configurations (`ConfigurationJson`) on success.

    pub fn read_from_file(pathname: &str) -> Vec<ConfigurationJson> {
        let content = fs::read_to_string(pathname).unwrap();
        let configuration_json_vec: Vec<ConfigurationJson> =
            serde_json::from_str(&content).unwrap();

        return configuration_json_vec;
    }
    /// Finds a neuron configuration by its identifier in the provided vector.
    ///
    /// # Arguments
    ///
    /// * `config_vec` - A vector of neuron configurations to search.
    /// * `target_config` - The identifier of the target neuron configuration.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to the found configuration (`&ConfigurationJson`)
    /// if it exists, or `None` otherwise.
    pub fn find_by_configuration(
        config_vec: &Vec<ConfigurationJson>,
        target_config: u32,
    ) -> Option<&ConfigurationJson> {
        config_vec
            .iter()
            .find(|&config| config.configuration == target_config)
    }
}

impl InputJson {
    pub fn read_input_from_file(pathname: &str) -> Vec<(u128, Vec<u128>)> {
        let content = fs::read_to_string(pathname).unwrap();
        // println!("content: {}", content);
        let input_json_vec: Vec<InputJson> = serde_json::from_str(&content).ok().unwrap();

        let mut input_vec: Vec<(u128, Vec<u128>)> = Vec::new();
        for i in input_json_vec {
            input_vec.push((i.neuron, i.spikes));
        }

        return input_vec;
    }
}

impl LayerWeightsJson {
    pub fn read_weights_from_file(pathname: &str) -> Vec<LayerWeightsJson> {
        let content = fs::read_to_string(pathname).unwrap();
        let weight_json_vec: Vec<LayerWeightsJson> = serde_json::from_str(&content).unwrap();

        // println!("WEIGHTS:");
        // for lw in &weight_json_vec {
        //     println!("{:?}", lw);
        // }

        return weight_json_vec;
    }

    pub fn find_by_layer(neuron_boxes: &[NeuronBox], target_layer: u32) -> Option<&NeuronBox> {
        return neuron_boxes
            .iter()
            .find(|&neuron_box| neuron_box.layer == target_layer);
    }
}

impl NeuronJson {
    /// Reads the neuron configuration from a JSON file and constructs a neural network.
    ///
    /// This function reads the configuration from the specified JSON file paths and
    /// creates a neural network based on the information provided.
    ///
    /// # Arguments
    ///
    /// * `layers_pathname` - A string slice representing the path to the layers JSON file.
    /// * `weights_pathname` - A string slice representing the path to the weights JSON file.
    /// * `configurations_pathname` - A string slice representing the path to the configurations JSON file.
    ///
    /// # Returns
    ///
    /// A `Result` containing the neural network (`NN<LeakyIntegrateFire>`) on success,
    /// or an error message (`String`) on failure
    pub fn read_from_file(
        layers_pathname: &str,
        weights_pathname: &str,
        configurations_pathname: &str,
    ) -> Result<NN<LeakyIntegrateFire>, String> {
        let content = fs::read_to_string(layers_pathname).unwrap();
        let neuron_json_list: Vec<NeuronJson> = serde_json::from_str(&content).ok().unwrap();
        let mut neuron_box_vec: Vec<NeuronBox> = Vec::new();
        // print!("{:?}", neuron_json_list);

        let configurations_list: Vec<ConfigurationJson> =
            ConfigurationJson::read_from_file(configurations_pathname);

        for nj in &neuron_json_list {
            let mut first_layer: Option<u32> = None;
            let mut last_layer: Option<u32> = None;

            let mut first_neuron: Option<u32> = None;
            let mut last_neuron: Option<u32> = None;

            // let config = Configuration::new(nj.v_rest, nj.v_reset, nj.v_threshold, nj.tau);
            let config_json =
                ConfigurationJson::find_by_configuration(&configurations_list, nj.configuration);
            let mut config: Configuration = Configuration::new(0.0, 0.0, 0.0, 0.0);

            match config_json {
                Some(c) => {
                    config = Configuration::new(c.v_rest, c.v_reset, c.v_threshold, c.tau);
                }
                None => {
                    println!("Error");
                }
            }

            if nj.layers.contains('-') {
                let parts: Vec<&str> = nj.layers.split('-').collect();

                if let Ok(first) = parts[0].parse::<u32>() {
                    first_layer = Some(first);
                }

                if let Ok(last) = parts[1].parse::<u32>() {
                    last_layer = Some(last);
                }
            } else {
                if let Ok(parsed_value) = nj.layers.parse::<u32>() {
                    first_layer = Some(parsed_value);
                    last_layer = Some(parsed_value);
                }
            }

            if nj.neurons.contains('-') {
                let parts: Vec<&str> = nj.neurons.split('-').collect();

                if let Ok(first) = parts[0].parse::<u32>() {
                    first_neuron = Some(first);
                }

                if let Ok(last) = parts[1].parse::<u32>() {
                    last_neuron = Some(last);
                }
            } else {
                if let Ok(parsed_value) = nj.neurons.parse::<u32>() {
                    first_neuron = Some(parsed_value);
                    last_neuron = Some(parsed_value);
                }
            }

            if let (Some(first_layer), Some(last_layer)) = (first_layer, last_layer) {
                for layer in first_layer..=last_layer {
                    // println!("LAYER: {}", layer);

                    if let (Some(first_neuron), Some(last_neuron)) = (first_neuron, last_neuron) {
                        for neuron in first_neuron..=last_neuron {
                            // println!(" Neuron: {}", neuron);
                            neuron_box_vec.push(NeuronBox {
                                layer,
                                position: neuron,
                                neuron: LifNeuron::from_conf(&config),
                            });
                        }
                    }
                }
            } else {
            }
        }

        neuron_box_vec.sort_by_key(|neuron_box| (neuron_box.layer, neuron_box.position));

        neuron_box_vec.dedup_by(|a, b| a.layer == b.layer && a.position == b.position);

        // println!("neuron box read, sorted and without duplicates");
        // for neuron_box in &neuron_box_vec {
        //     println!(" {:?}", neuron_box)
        // }

        let weights_from_file = LayerWeightsJson::read_weights_from_file(weights_pathname);
        let mut layer_neurons: Vec<LifNeuron> = Vec::new();
        let mut nn = NN::<LeakyIntegrateFire>::new();
        let mut current_layer: usize = 0;

        for neuron_box in &neuron_box_vec {
            // println!("weights {:?}",weights_from_file[current_layer]);
            // println!("layer {:?}",current_layer);

            if current_layer == neuron_box.layer as usize {
                layer_neurons.push(neuron_box.neuron.clone());
            } else {
                let current_layer_weights = &weights_from_file[current_layer];

                // Create the layer
                // println!("I neuroni del layer ? sono {:?}:",layer_neurons.clone());
                // println!("I pesi del layer ? sono ({}x{}) {:?}:",current_layer_weights.input_weights.cols,current_layer_weights.input_weights.rows,current_layer_weights.input_weights.data.clone());
                nn = nn
                    .clone()
                    .layer(
                        layer_neurons.clone(),
                        DMatrix::from_vec(
                            current_layer_weights.input_weights.rows,
                            current_layer_weights.input_weights.cols,
                            current_layer_weights.input_weights.data.clone(),
                        ),
                        DMatrix::from_vec(
                            current_layer_weights.intra_weights.rows,
                            current_layer_weights.intra_weights.cols,
                            current_layer_weights.intra_weights.data.clone(),
                        ),
                    )
                    .unwrap();

                current_layer += 1;
                layer_neurons.clear();

                // First iteration of the next layer
                layer_neurons.push(neuron_box.neuron.clone());
            }
        }

        if !layer_neurons.is_empty() {
            let current_layer_weights = &weights_from_file[current_layer];
            // println!("I neuroni del layer ? sono {:?}:",layer_neurons.clone());
            // println!("I pesi del layer ? sono ({}x{}) {:?}:",current_layer_weights.input_weights.cols,current_layer_weights.input_weights.rows,current_layer_weights.input_weights.data.clone());

            nn = nn.clone()
                .layer(
                    layer_neurons.clone(),
                    DMatrix::from_vec(
                        current_layer_weights.input_weights.rows,
                        current_layer_weights.input_weights.cols,
                        current_layer_weights.input_weights.data.clone(),
                    ),
                    DMatrix::from_vec(
                        current_layer_weights.intra_weights.rows,
                        current_layer_weights.intra_weights.cols,
                        current_layer_weights.intra_weights.data.clone(),
                    ),
                )
                .expect("Error in layer");
        } else {
            return Err("error".to_string());
        }
        return Ok(nn);
    }
}
/// Represents the configuration for resilience read from a JSON file.
#[derive(Debug, Deserialize)]
pub struct ResilienceJson {
    components: Vec<String>,
    stuck: String,
    times: u32,
}
impl ResilienceJson {
    /// Reads the resilience configuration from a JSON file and returns a `Result`.
    ///
    /// # Arguments
    ///
    /// * `pathname` - A string slice representing the path to the JSON file.
    ///
    /// # Returns
    ///
    /// A `Result` containing `ResilienceJson` on success or an error message `String` on failure.

    pub fn read_from_file(pathname: &str) -> Result<ResilienceJson, String> {
        let content = fs::read_to_string(pathname).unwrap();
        let resilience_configuration_json: ResilienceJson =
            serde_json::from_str(&content).ok().unwrap();

        return Ok(resilience_configuration_json);
    }
    /// Converts `ResilienceJson` into `Resilience`.
    ///
    /// # Returns
    ///
    /// A `Result` containing `Resilience` on success or an error message `String` on failure.

    pub fn to_resilience(self) -> Result<Resilience, String> {
        let stuck_type = match self.stuck.to_lowercase().as_str() {
            "stuck_at_0" | "zero" | "z" | "0" => Ok(Stuck::Zero),
            "stuck_at_1" | "one" | "o" | "1" => Ok(Stuck::One),
            "transient_bit" | "transient" | "t" | "2" => Ok(Stuck::Transient),
            _ => Err(format!("Invalid stuck type: {}", self.stuck)),
        }?;

        Ok(Resilience {
            components: self.components,
            stuck_type,
            times: self.times as u128,
        })
    }
}


/// Represents the paths.
#[derive(Debug, Deserialize, Clone)]
pub struct PathsJson {
    pub configurations: String,
    pub input_spikes: String,
    pub layers: String,
    pub resilience: String,
    pub weights: String
}

impl PathsJson {
    /// Reads the paths from a JSON file and returns a `Result`.
    ///
    /// # Arguments
    ///
    /// * `pathname` - A string slice representing the path to the JSON file.
    ///
    /// # Returns
    ///
    /// A `Result` containing `PathsJson` on success or an error message `String` on failure.

    pub fn read_from_file(pathname: &str) -> Result<PathsJson, String> {
        match fs::read_to_string(pathname) {
            Ok(data) => {
                match serde_json::from_str(&data) {
                    Ok(paths_json) => Ok(paths_json),
                    Err(_) => Err("Errore durante la deserializzazione del file JSON.".to_string()),
                }
            }
            Err(ref e) if e.kind() == ErrorKind::NotFound => {
                let default_content = r#"{
  "configurations": "src/configuration/configurations.json",
  "input_spikes": "src/configuration/input_spikes.json",
  "layers": "src/configuration/layers.json",
  "resilience": "src/configuration/resilience.json",
  "weights": "src/configuration/weights.json"
}"#;
                match File::create(pathname) {
                    Ok(mut file) => {
                        // Scriviamo il contenuto di default nel file appena creato.
                        if let Err(e) = file.write_all(default_content.as_bytes()) {
                            return Err(format!("Errore durante la scrittura nel file: {}", e));
                        }
                        Err("File dei path creato. Si prega di compilare con i path appropriati.".to_string())
                    }
                    Err(e) => Err(format!("Errore durante la creazione del file: {}", e)),
                }
            }
            _ => Err(format!("Error: unexpected"))
        }
    }
}
