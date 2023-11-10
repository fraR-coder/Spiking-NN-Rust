use spiking_nn_resilience::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
use spiking_nn_resilience::*;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct NeuronJson {
    neuron_number: u32,
    layer: u32,
    v_rest: f64, 
    v_reset: f64, 
    v_threshold: f64, 
    tau: f64
}

#[derive(Debug, Deserialize)]
pub struct WeightJson {
    rows: u32,
    cols: u32,
    data: Vec<f64>
}

#[derive(Debug, Deserialize)]
pub struct LayerWeightsJson {
    layer: u32,
    input_weights: WeightJson,
    intra_weights: WeightJson
}

impl NeuronJson{
    pub fn read_from_file(pathname: &str) -> Vec<NeuronJson>{
        let content = std::fs::read_to_string(pathname).unwrap();
        let mut neuronjson_list: Vec<NeuronJson> = serde_json::from_str(&content).unwrap();

        // Ordering the neurons...
        neuronjson_list.sort_by(|a, b| {
            // Ordering by layer
            let cmp_index1 = a.layer.cmp(&b.layer);
            
            // Ordering by neuron number
            if cmp_index1 == std::cmp::Ordering::Equal {
                a.neuron_number.cmp(&b.neuron_number)
            } else {
                cmp_index1
            }
        });

        // Network initialization 
        let nn = NN::<LeakyIntegrateFire>::new();
        let mut current_layer = 0;

        let mut layer_lif_neurons: Vec<LifNeuron> = Vec::new();
        for neuron in &neuronjson_list {
            
            if current_layer == neuron.layer {
                let config = Configuration::new(neuron.v_rest, neuron.v_reset, neuron.v_threshold, neuron.tau);
                layer_lif_neurons.push(LifNeuron::from_conf(&config));
            }
            else {
                // Import weights
                // ...

                // Create the layer
                // nn.layer(layer_lif_neurons, ...)

                current_layer += 1;
                layer_lif_neurons.clear();

                // First iteration of the next layer
                let config = Configuration::new(neuron.v_rest, neuron.v_reset, neuron.v_threshold, neuron.tau);
                layer_lif_neurons.push(LifNeuron::from_conf(&config));
            }
            
        }

        return neuronjson_list
    }
}
