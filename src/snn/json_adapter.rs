use crate::snn::model::lif::*;
use crate::NN;

use serde::Deserialize;

use nalgebra::DMatrix;
use crate::snn::model::Stuck;
use crate::snn::resilience::Resilience;

#[derive(Debug, Deserialize)]
pub struct NeuronJson {
    neurons: String,
    layers: String,
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64
}


#[derive(Debug)]
pub struct NeuronBox {
    layer: u32,
    position: u32,
    neuron: LifNeuron
}

#[derive(Debug, Deserialize)]
pub struct WeightJson {
    rows: usize,
    cols: usize,
    data: Vec<f64>
}

#[derive(Debug, Deserialize)]
pub struct LayerWeightsJson {
    layer: u32,
    input_weights: WeightJson,
    intra_weights: WeightJson
}

#[derive(Debug, Deserialize)]
pub struct InputJson {
    neuron: u128,
    spikes: Vec<u128>
}

impl InputJson {
    pub fn read_input_from_file(pathname: &str) -> Vec<(u128, Vec<u128>)> {
        let content = std::fs::read_to_string(pathname).unwrap();
        // println!("content: {}", content);
        let mut input_json_vec: Vec<InputJson> = serde_json::from_str(&content).ok().unwrap();

        let mut input_vec: Vec<(u128, Vec<u128>)> = Vec::new();
        for i in input_json_vec {
            input_vec.push((i.neuron, i.spikes));
        }

        return input_vec
    }
}

impl LayerWeightsJson {
    pub fn read_weights_from_file(pathname: &str) -> Vec<LayerWeightsJson> {
        let content = std::fs::read_to_string(pathname).unwrap();
        let mut weight_json_vec: Vec<LayerWeightsJson> = serde_json::from_str(&content).unwrap();

        println!("WEIGHTS:");
        for lw in &weight_json_vec {
            println!("{:?}", lw);
        }


        return weight_json_vec;
    }

    pub fn find_by_layer(neuron_boxes: &[NeuronBox], target_layer: u32) -> Option<&NeuronBox> {
        return neuron_boxes.iter().find(|&neuron_box| neuron_box.layer == target_layer)
    }
}

impl NeuronJson{

    pub fn read_from_file(layers_pathname: &str, weights_pathname: &str) -> Result<NN<LeakyIntegrateFire>,String>{
        let content = std::fs::read_to_string(layers_pathname).unwrap();
        let neuron_json_list: Vec<NeuronJson> = serde_json::from_str(&content).ok().unwrap();
        let mut neuron_box_vec: Vec<NeuronBox> = Vec::new();
        // print!("{:?}", neuron_json_list);

        for nj in &neuron_json_list {

            let mut first_layer: Option<u32> = None;
            let mut last_layer: Option<u32> = None;

            let mut first_neuron: Option<u32> = None;
            let mut last_neuron: Option<u32> = None;

            let config = Configuration::new(nj.v_rest, nj.v_reset, nj.v_threshold, nj.tau);

            // Verifico se c'è una sequenza di layer
            if nj.layers.contains('-') {
                let parts: Vec<&str> = nj.layers.split('-').collect();

                // Converto in u32 gli estremi
                if let Ok(first) = parts[0].parse::<u32>() {
                    first_layer = Some(first);
                }

                if let Ok(last) = parts[1].parse::<u32>() {
                    last_layer = Some(last);
                }
            }
            else {
                if let Ok(parsed_value) = nj.layers.parse::<u32>() {
                    first_layer = Some(parsed_value);
                    last_layer = Some(parsed_value);
                }
            }

            // Verifico se c'è una sequenza di neuroni
            if nj.neurons.contains('-') {
                let parts: Vec<&str> = nj.neurons.split('-').collect();

                // Converto in u32 gli estremi
                if let Ok(first) = parts[0].parse::<u32>() {
                    first_neuron = Some(first);
                }

                if let Ok(last) = parts[1].parse::<u32>() {
                    last_neuron = Some(last);
                }
            }
            else {
                if let Ok(parsed_value) = nj.neurons.parse::<u32>() {
                    first_neuron = Some(parsed_value);
                    last_neuron = Some(parsed_value);
                }
            }

            // Iterazione da first_layer a last_layer
            if let (Some(first_layer), Some(last_layer)) = (first_layer, last_layer) {

                for layer in first_layer..=last_layer {

                    // println!("LAYER: {}", layer);

                    // Iterazione da first_neuron a last_neuron
                    if let (Some(first_neuron), Some(last_neuron)) = (first_neuron, last_neuron) {

                        for neuron in first_neuron..=last_neuron {
                            // println!(" Neuron: {}", neuron);
                            neuron_box_vec.push(NeuronBox{layer, position: neuron, neuron: LifNeuron::from_conf(&config)});
                        }
                    }
                }

            }
            else {
                println!("Valori non validi per l'iterazione");
            }

        }

        // Ordino i neuron box, in modo da averli raggruppati per layer e position
        neuron_box_vec.sort_by_key(|neuron_box| (neuron_box.layer, neuron_box.position));

        // Elimino i possibili duplicati
        neuron_box_vec.dedup_by(|a, b| a.layer == b.layer && a.position == b.position);

        println!("\nNEURON BOX LETTI, ORDINATI E SENZA DUPLICATI");
        for neuron_box in &neuron_box_vec {
            println!(" {:?}", neuron_box)
        }

        let weights_from_file = LayerWeightsJson::read_weights_from_file(weights_pathname);

        let mut layer_neurons: Vec<LifNeuron> = Vec::new();
        let mut nn = NN::<LeakyIntegrateFire>::new();
        let mut current_layer: usize = 0;

        for neuron_box in &neuron_box_vec {
            if current_layer == neuron_box.layer as usize {
                layer_neurons.push(neuron_box.neuron.clone());
            } else {
                let current_layer_weights = &weights_from_file[current_layer];

                // Create the layer
                nn = nn.clone().layer(
                    layer_neurons.clone(), // Clono il vettore layer_neurons
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
                ).unwrap();

                current_layer += 1;
                layer_neurons.clear();

                // First iteration of the next layer
                layer_neurons.push(neuron_box.neuron.clone());
            }
        }

        // Aggiungo l'ultimo layer
        if !layer_neurons.is_empty() {
            let current_layer_weights = &weights_from_file[current_layer];
            nn = nn.clone().layer(
                layer_neurons.clone(), // Clono il vettore layer_neurons
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
            ).expect("Error in layer");
        }else {
            return Err("errore".to_string());
        }
        return Ok(nn)
    }
}
#[derive(Debug, Deserialize)]
pub struct ResilienceJson {
    components: Vec<String>,
    stuck: String,
    times: u32
}
impl ResilienceJson {
    pub fn read_from_file(pathname: &str) -> Result<ResilienceJson, String> {
        let content = std::fs::read_to_string(pathname).unwrap();
        let resilienceConfigurationJson: ResilienceJson = serde_json::from_str(&content).ok().unwrap();

        return Ok(resilienceConfigurationJson);
    }
    pub fn to_resilience(self) -> Result<Resilience, String> {
        let stuck_type = match self.stuck.to_lowercase().as_str() {
            ("stuck_at_0" | "zero" | "z" | "0")  => Ok(Stuck::Zero),
            ("stuck_at_1" | "one" | "o" | "1") => Ok(Stuck::One),
            ("transient_bit" | "transient" | "t" | "2") => Ok(Stuck::Transient),
            _ => Err(format!("Invalid stuck type: {}", self.stuck)),
        }?;

        Ok(Resilience {
            components: self.components,
            stuck_type,
            times: self.times as u128,
        })
    }
}
