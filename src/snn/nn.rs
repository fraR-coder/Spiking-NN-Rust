//! # Spiking Neural Network Library
//!
//! This Rust library provides a framework for spiking neural networks (SNNs) with support for resilience mechanisms.
//! The library defines a generic `NN` (Neural Network) structure that represents a collection of layers.
//!
//! ## Overview
//!
//! The `NN` structure is generic over a type `M` that must implement the `Model` trait. The library provides support
//! for creating and simulating SNNs with various neuron models, such as Leaky Integrate-and-Fire (LIF) neurons.
//!
//! ## Usage
//!
//! The library includes the following main components:
//!
//! - **Neural Network Structure (`NN`):** Represents a collection of layers.
//! - **Layer Structure (`Layer<M>`):** Represents an individual layer in the neural network.
//! - **Spike Structure (`Spike`):** Represents a spike event with timestamp, layer index, and neuron index.
//!
//! ## Neuron Model
//!
//! The type `M` in the `NN` structure must implement the `Model` trait. The trait includes methods like `handle_spike`
//! for processing incoming spikes and updating neuron states. Specific neuron models, such as Leaky Integrate-and-Fire,
//! can be implemented by providing the necessary functionalities.
//!
//! ## Creating a Neural Network
//!
//! The library supports the creation of a neural network with multiple layers using the `layer` method of the `NN` structure.
//! Each layer is defined by the number of neurons, input weights, and intra-layer weights.
//!
//! ```rust
//! use spiking_nn_resilience::{NN, Model};
//! use nalgebra::DMatrix;
//!
//! // Define a neuron model (e.g., Leaky Integrate-and-Fire)
//! struct LeakyIntegrateFire;
//!
//! impl Model for LeakyIntegrateFire {
//!     // Implement required methods for the neuron model
//! }
//!
//! // Create a neural network with LIF neurons
//! let nn = NN::<LeakyIntegrateFire>::new()
//!     .layer(
//!         /* neurons */ vec![/* neuron instances */],
//!         /* input_weights */ DMatrix::from_vec(1, 1, vec![1.0]),
//!         /* intra_weights */ DMatrix::from_vec(1, 1, vec![0.0]),
//!     )
//!     .expect("Error creating the neural network.");
//! ```
//!
//! ## Solving the Neural Network
//!
//! The `NN` structure provides methods for solving the SNN given input spikes. The `solve_multiple_vec_spike` method takes a
//! vector of tuples representing neuron indices and their corresponding vectors of spikes. The simulation duration is also specified.
//!
//! ```rust
//! // Assuming 'nn' is an instance of the NN structure
//!
//! let input = vec![(0, vec![1, 2, 3]), (1, vec![2, 4, 6])];
//! let duration = 6;
//!
//! let shared_output = nn.clone().solve_multiple_vec_spike(input);
//!
//! // Access the results using the 'shared_output' Arc.
//! ```
//!
//! ## Resilience Mechanisms
//!
//! The library includes support for resilience mechanisms. Specifics of the resilience handling are not provided in this documentation,
//! but the `ResilienceJson` module suggests a mechanism for reading resilience configurations from files and converting them into resilience objects.
//!


use crate::snn::layer::Layer;
use crate::Model;
use nalgebra::DMatrix;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use super::Spike;

/// Neural Network (`NN`) structure representing a collection of layers.
/// The neural network is defined by the type parameter `M`, which must implement
/// the `Model` trait.
#[derive(Clone)]
pub struct NN<M: Model + Clone + 'static> {
    /// All the sorted layers of the neural network
    pub layers: Vec<Layer<M>>,
}

impl<M: Model + Clone> NN<M> {
    /// Creates a new empty neural network.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }
    /// Adds a new layer to the neural network with specified neurons, input weights,
    /// and intra-layer weights.
    ///
    /// # Arguments
    ///
    /// * `neurons` - Vector of neurons for the new layer.
    /// * `input_weights` - Matrix representing input weights. Should have dimensions
    ///   compatible with the number of neurons in the previous layer.
    /// * `intra_weights` - Matrix representing intra-layer weights. Should have dimensions
    ///   compatible with the number of neurons in the current layer.
    ///
    /// # Returns
    ///
    /// * `Result<Self, String>` - Result containing the updated neural network or an error message.
    ///
    /// # Examples
    ///
    /// ```
    /// use spiking_nn_resilience::snn::layer::Layer;
    /// use spiking_nn_resilience::{Model, NN};
    /// use nalgebra::{DMatrix, DVector};
    /// use std::sync::{mpsc, Arc, Mutex};
    /// use std::thread;
    /// use spiking_nn_resilience::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
    /// let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0);
    /// let nn = NN::<LeakyIntegrateFire>::new()
    ///     .layer(
    ///         vec![LifNeuron::from_conf(&config_0)],
    ///     DMatrix::from_vec(1, 1, vec![1.0]),
    ///     DMatrix::from_vec(1, 1, vec![0.0]))
    ///     .expect("Error creating the neural network.");
    pub fn layer(
        mut self,
        neurons: Vec<M::Neuron>,
        input_weights: DMatrix<f64>,
        intra_weights: DMatrix<f64>,
    ) -> Result<Self, String> {
        let len_last_layer = self.layers.last().map(|l| l.neurons.len()).unwrap_or(0);
        let n = neurons.len();

        // Check layer len not zero
        if n == 0 {
            return Err("The number of neurons should be at least 1".to_string());
        }

        // Check size compatibilities
        if intra_weights.len() != n * n {
            return Err("Incompatible intra weight matrix".to_string());
        }

        if input_weights.len()
            != (if len_last_layer == 0 {
                n * n
            } else {
                len_last_layer * n
            })
        {
            return Err("Incompatible input weight matrix".to_string());
        }

        // Finally, insert layer into nn
        let new_layer = Layer {
            neurons,
            input_weights,
            intra_weights,
        };
        self.layers.push(new_layer);

        Ok(self)
    }
    pub fn get_num_layers(&self) -> usize {
        return self.layers.len();
    }
    /*
       Ho un vettore di spike iniziali.
       Inizialmente si crea il primo thread responsabile di gestire il primo vettore di spike e quindi il primo
       layer. Successivamente si crea il secondo thread e poi il terzo ecc.
       Quando ha finito la computazione, ciascun thread invia il vettore di spike che ha generato il suo layer al thread
       che si occupa del livello successivo. Quando l'ultimo spike di input è stato inviato e usato,
       i thread si eliminano uno ad uno.
    */
    // inizialmente la funzione riceve un vettore ordinato di u128 (ovvero istanti di tempo in cui si applica
    // lo spike al primo layer. Successivamente si potrebbe cambiare implementazione...
    /// Solves the SNN given a single vector of spikes -> (Vec<u128<).
    /// Each spike is given to all neurons of the first layer, so all
    /// of them have a global visibility of the input
    // pub fn solve_single_vec_spike(&mut self, input: Vec<u128>) {
    //     // println!("Enter solve");
    //     let mut index_input: usize = 0;
    //     let duration = match input.last().clone() {
    //         //why cloning
    //         Some(value) => value.clone(),
    //         None => 0,
    //     };
    //     // one channel for every layer
    //     let num_layers = self.get_num_layers();
    //     let mut channel_tx = vec![];
    //     let mut channel_rx = vec![];
    //     let (mut first_tx, _) = mpsc::channel();
    //     for i in 0..num_layers {
    //         let (tx, rx): (mpsc::Sender<Vec<Spike>>, mpsc::Receiver<Vec<Spike>>) = mpsc::channel();
    //         if i == 0 {
    //             first_tx = tx.clone();
    //         }
    //         channel_tx.push(tx);
    //         channel_rx.push(rx);
    //     }
    //
    //     // Creazione dei thread
    //     let mut handles = vec![];
    //     for layer_idx in 0..channel_tx.len() {
    //         let next_tx = if layer_idx < num_layers - 1 {
    //             Some(channel_tx[layer_idx + 1].clone())
    //         } else {
    //             None
    //         };
    //         let rx = channel_rx.remove(0);
    //         let mut layers = self.layers.clone();
    //         let thread_name = format!("layer_{}", layer_idx);
    //         let handle = thread:://Builder::new()
    //             /*.name(thread_name) // Imposta il nome del thread*/
    //             spawn(move || {
    //             // il vettore di spike da passare al thread successivo
    //             let mut layer_output: Vec<Spike> = vec![]; // Output del layer (Spike generati)
    //
    //             for ts in layer_idx as u128..duration + layer_idx as u128 {
    //                 // Ricezione degli spike dal layer precedente
    //                 let input_spike = match rx.recv() {
    //                     Ok(input_spike) => input_spike,
    //                     Err(_err) => {
    //                         println!("Error receiving the vector of spikes, layer: {}", layer_idx);
    //                         vec![]
    //                     }
    //                 };
    //                 // eseguo i calcoli per aggiornare le tensioni di membrana e riempio il layer_output di spike
    //                 for neuron_idx in 0..layers[layer_idx].num_neurons() as u128 {
    //                     let mut sum: f64 = 0.0;
    //                     let input_spike_tmp = input_spike.clone();
    //                     for spike in input_spike_tmp.into_iter() {
    //                         //let neuron=layers[layer_idx].get_neuron(neuron_idx as usize);
    //
    //                         //sum_whith_injection(neuron_idx,layers[layer_idx],spike);
    //                         sum += 1.0
    //                             * layers[layer_idx].input_weights
    //                                 [[neuron_idx as usize][spike.neuron_id]]; // da reimplementare in una funzione a parte con gli stuck
    //                     }
    //                     let res = M::handle_spike(
    //                         layers[layer_idx]
    //                             .get_neuron_mut(neuron_idx as usize)
    //                             .unwrap(),
    //                         sum,
    //                         ts,
    //                     );
    //                     if res == 1. {
    //                         layer_output.push(Spike::new(ts, layer_idx, neuron_idx as usize))
    //                     }
    //                 }
    //
    //                 // Invio dei nuovi spike al layer successivo (se presente)
    //                 if let Some(next_tx) = &next_tx {
    //                     next_tx
    //                         .send(layer_output.clone())
    //                         .expect("Error sending the vector of spikes");
    //                 } else {
    //                     for s in layer_output.iter() {
    //                         // println!("Output: {} (thread: {})", s.clone(), thread_name);
    //                     }
    //                 }
    //                 layer_output = vec![];
    //             } // end for
    //
    //             // Salvataggio degli spike generati dal layer nell'array
    //             //let mut layer_data = layer_clone[layer_idx].lock().unwrap();
    //             //layer_data.extend(layer_output);
    //         });
    //
    //         handles.push(handle);
    //     }
    //     for ts in 0..duration {
    //         let spike = match input.get(index_input) {
    //             Some(value) => value.clone(),
    //             None => 0,
    //         };
    //         index_input += 1;
    //         for _ in ts..spike {
    //             first_tx
    //                 .send(vec![])
    //                 .expect("Error sending the vector of spikes");
    //         }
    //         let mut vec_of_spikes = vec![];
    //         for _ in 0..self.layers[0].num_neurons() {
    //             vec_of_spikes.push(Spike::new(spike, 0, 0));
    //         }
    //         first_tx
    //             .send(vec_of_spikes)
    //             .expect("Error sending the vector of spikes");
    //     }
    //
    //     // Attendo il completamento di tutti i thread
    //     for handle in handles {
    //         handle.join().expect("kk"); //.expect("Failed to join a thread");
    //     }
    // }

    /// Solves the SNN given a vector of Tuples (neuron_id, vectors of spikes) -> (Vec<(u128, Vec<u128>)>).
    /// Each spike is referred to a single input neuron, providing reduced visibility to neurons in each layer.
    ///
    /// # Arguments
    ///
    /// * `input` - Vector of tuples where each tuple represents the neuron_id and the corresponding vector of spikes.
    /// * `duration` - The duration of the simulation.
    ///
    /// # Returns
    ///
    /// * `Arc<Mutex<Vec<(u128, Vec<u128>)>>>` - Arc-wrapped Mutex-protected vector containing tuples of neuron_id and generated spikes.
    ///   Each tuple corresponds to the output of a neuron in the final layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use spiking_nn_resilience::{NN, Model};
    /// use spiking_nn_resilience::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
    /// use nalgebra::DMatrix;
    /// use std::sync::{Arc, Mutex};
    /// let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0);
    /// let nn = NN::<LeakyIntegrateFire>::new()
    ///     .layer(
    ///         vec![LifNeuron::from_conf(&config_0)],
    ///     DMatrix::from_vec(1, 1, vec![1.0]),
    ///     DMatrix::from_vec(1, 1, vec![0.0]))
    ///     .expect("Error creating the neural network.");
    /// // Add layers to the neural network
    /// // ...
    ///
    /// let input = vec![(0, vec![1, 2, 3])];
    /// let duration = 3;
    ///
    /// let shared_output = nn.clone().solve_multiple_vec_spike(input);
    ///
    /// // Access the results using the shared_output Arc.
    pub fn solve_multiple_vec_spike(
        &mut self,
        input: Vec<(u128, Vec<u128>)>,
    ) -> Arc<Mutex<Vec<(u128, Vec<u128>)>>> {
        //println!("Enter solve multiple vec spike");
        let mut index_input: usize = 0;
        // creo tanti canali quanti sono i layer
        let num_layers = self.get_num_layers();

        let shared_output = Arc::new(Mutex::new(Vec::<(u128, Vec<u128>)>::new()));
        for i in 0..self.layers.last().unwrap().num_neurons() {
            shared_output.lock().unwrap().push((i as u128, vec![]));
        }

        // let mut output:Vec<(u128, Vec<u128>)> = vec![];
        let mut channel_tx = vec![];
        let mut channel_rx = vec![];
        let (mut first_tx, _) = mpsc::channel();
        for i in 0..num_layers {
            let (tx, rx): (mpsc::Sender<Vec<Spike>>, mpsc::Receiver<Vec<Spike>>) = mpsc::channel();
            if i == 0 {
                first_tx = tx.clone();
            }
            channel_tx.push(tx);
            channel_rx.push(rx);
        }

        // Creazione dei thread
        let mut handles = vec![];
        let input_spikes = Spike::vec_of_all_spikes(input);
        let duration = input_spikes.last().clone().unwrap().ts as usize;
        for layer_idx in 0..num_layers {
            let next_tx = if layer_idx < num_layers - 1 {
                Some(channel_tx[layer_idx + 1].clone())
            } else {
                None
            };
            let rx = channel_rx.remove(0);
            let mut layers = self.layers.clone();
            let _thread_name = format!("layer_{}", layer_idx);
            let output_clone = shared_output.clone();
            let handle = thread:://Builder::new()
            /*.name(thread_name) // Imposta il nome del thread*/
            spawn(move || {
                // il vettore di spike da passare al thread successivo
                let mut layer_output: Vec<Spike> = vec![]; // Output del layer (Spike generati)
                let mut ts: u128 = 0;
                // let mut counter: u128 = 0;
                let mut neuron_counters: Vec<u128> = vec![];
                for _ in 0..layers[layer_idx].num_neurons() {
                    neuron_counters.push(0);
                }
                while ts <= duration as u128 + layer_idx as u128 {
                    // Ricezione degli spike dal layer precedente
                    let input_spike = match rx.recv() {
                        Ok(input_spike) => input_spike,
                        Err(_err) => {
                            println!("Error receiving the vector of spikes, layer: {}", layer_idx);
                            vec![]
                        }
                    };

                    if input_spike.is_empty() {
                        if let Some(next_tx) = &next_tx {
                            next_tx
                                .send(vec![])
                                .expect("Error sending the vector of spikes");
                            break;
                        } else {
                            //println!("counter: {}",counter);
                            //println!("neuron counters: {:?}",neuron_counters);

                            break;
                        }
                    }

                    //println!("input_spikes: {:?} (Thread: {})",input_spike,thread_name);
                    //println!("input ts: {}, layer_idx: {}",input_spike[0].ts.clone(), layer_idx.clone());
                    //ts=input_spike[0].ts-layer_idx as u128;
                    //println!("Receive vec: {:?}",input_spike);
                    // eseguo i calcoli per aggiornare le tensioni di membrana e riempio il layer_output di spike
                    for neuron_idx in 0..layers[layer_idx].num_neurons() as u128 {
                        let sum: f64;
                        let input_spike_tmp = input_spike.clone();
                        let mut s = 0;
                        if !input_spike_tmp.is_empty() {
                            s = input_spike_tmp.clone().get(0).unwrap().ts;
                        }

                        // println!("\n\n\n\n input spike: {:?}, neuron_idx: {}, matrix: {}",input_spike_tmp.clone(), neuron_idx, &layers[layer_idx].input_weights);

                        //do the sum considering the possible errors
                        sum = Self::calculate_sum(input_spike_tmp, &layers[layer_idx], neuron_idx);

                        /*for spike in input_spike_tmp.into_iter(){
                            sum += 1.0*layers[layer_idx].input_weights[(spike.neuron_id, neuron_idx as usize)]; // da reimplementare in una funzione a parte con gli stuck
                        }
                        */
                        let res = M::handle_spike(
                            layers[layer_idx]
                                .get_neuron_mut(neuron_idx as usize)
                                .unwrap(),
                            sum,
                            s,
                        );
                        if res == 1.0 {
                            layer_output.push(Spike::new(
                                input_spike[0].ts + 1,
                                layer_idx,
                                neuron_idx as usize,
                            ))
                        }
                    }

                    // Invio dei nuovi spike al layer successivo (se presente)
                    if let Some(next_tx) = &next_tx {
                        if !layer_output.is_empty() {
                            //println!("vec in output: {:?} sent from thread:{}",layer_output.clone(),thread_name);
                            next_tx
                                .send(layer_output.clone())
                                .expect("Error sending the vector of spikes");
                            layers[layer_idx].update_layer_cycle(&layer_output);
                        }
                    } else {
                        for s in layer_output.iter() {
                            if let Some((_, v)) = output_clone
                                .lock()
                                .unwrap()
                                .iter_mut()
                                .find(|(k, _)| *k == s.neuron_id as u128)
                            {
                                v.push(s.ts);
                            }
                            //println!("final Output: {} (thread: {})",s.clone(),thread_name);
                            // counter += 1;
                            neuron_counters[s.neuron_id] += 1;
                        }
                    }
                    layer_output = vec![];
                    ts += 1;
                } // end while

                // Salvataggio degli spike generati dal layer nell'array
                //let mut layer_data = layer_clone[layer_idx].lock().unwrap();
                //layer_data.extend(layer_output);
            }); // end thread
            handles.push(handle);
        }
        let mut ts: usize = 0;
        while ts <= duration {
            let spike = match input_spikes.get(index_input) {
                Some(value) => value.clone(),
                None => Spike::new(0, 0, 0),
            };
            ts = spike.ts as usize;
            let mut vec_of_spikes = vec![];
            // println!("input spikes to starting: {:?}",input_spikes);
            for _ in 0..self.layers[0].num_neurons() {
                let s = match input_spikes.get(index_input) {
                    Some(value) => value.clone(),
                    None => Spike::new(0, 0, 0),
                };
                if s.ts == spike.ts {
                    vec_of_spikes.push(s);
                    index_input += 1;
                    //println!("Emit spike: {}",spike);
                } else {
                    break;
                }
            }
            first_tx
                .send(vec_of_spikes.clone())
                .expect("Error sending the vector of spikes");
            //println!("Sent vec: {:?} to T0",vec_of_spikes);
            ts += 1;
        }
        first_tx
            .send(vec![])
            .expect("Error sending the vector of spikes");

        // Attendo il completamento di tutti i thread
        for handle in handles {
            handle.join().expect("Ok"); //.expect("Failed to join a thread");
        }
        return shared_output;
    }

    /// Calculates the sum of weighted inputs based on the received spikes and the layer's configuration.
    ///
    /// # Arguments
    ///
    /// * `input_spike_tmp` - Vector of spikes received by the neuron.
    /// * `layer` - Reference to the layer configuration.
    /// * `neuron_idx` - Index of the neuron within the layer.
    ///
    /// # Returns
    ///
    /// * `f64` - The calculated sum of weighted inputs.
    ///
    /// # Examples
    ///
    /// ```
    /// use spiking_nn_resilience::{NN, Model, Layer};
    /// use nalgebra::DMatrix;
    /// use spiking_nn_resilience::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
    /// use spiking_nn_resilience::snn::Spike;
    /// let mut nn = NN::<LeakyIntegrateFire>::new();
    /// // Add layers to the neural network
    /// // ...
    /// let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0);
    /// let nn = NN::<LeakyIntegrateFire>::new()
    ///     .layer(
    ///         vec![LifNeuron::from_conf(&config_0)],
    ///     DMatrix::from_vec(1, 1, vec![1.0]),
    ///     DMatrix::from_vec(1, 1, vec![0.0]));
    /// let neuron_idx = 0;
    /// let input_spike_tmp = vec![Spike::new(1, 0, 0), Spike::new(2, 0, 0)];
    ///
    /// let sum = NN::calculate_sum(input_spike_tmp, nn.unwrap().layers.last().unwrap(), neuron_idx as u128);
    /// ```
    pub fn calculate_sum(input_spike_tmp: Vec<Spike>, layer: &Layer<M>, neuron_idx: u128) -> f64 {
        let neuron = layer.get_neuron(neuron_idx as usize).unwrap();
        let inputs_to_sum: Vec<f64> = input_spike_tmp
            .into_iter()
            .map(|spike| layer.input_weights[(spike.neuron_id, neuron_idx as usize)])
            .collect();

        if let Some(mut heap_vec) = M::get_heap(neuron) {
            heap_vec.sum_all(&inputs_to_sum)
        } else {
            inputs_to_sum.iter().sum()
        }
    }

    // pub fn solve_single_thread(mut self, input: Vec<u128>) -> Vec<(u128, Vec<f64>)> {
    //     let n_neurons_0 = self.layers[0].num_neurons();
    //     let mut result_vec: Vec<(u128, Vec<f64>)> = Vec::new();
    //
    //     for ts_i in input {
    //         let mut ts = ts_i;
    //         //matrice 1xn
    //         let mut spike_mat = DVector::from_element(n_neurons_0, 1.0).transpose();
    //         for (layer_i, layer) in self.layers.iter_mut().enumerate() {
    //             let mat = &layer.input_weights; //matrice nxn colonna0
    //             let weighted_matrix = &spike_mat * mat; //matrice 1xn colonna 0 neruone0 ecc..
    //
    //             ts += 1;
    //             let mut real_spike_vec = Vec::new();
    //
    //             let mut v_spike = Vec::new();
    //             for (index, neuron) in layer.neurons.iter_mut().enumerate() {
    //                 let weighted_input_val = weighted_matrix[index];
    //                 // println!(
    //                 //     "layer: {} neuron:{} ts:{} val:{}",
    //                 //     layer_i, index, ts, weighted_input_val
    //                 // );
    //
    //                 let res = M::handle_spike(neuron, weighted_input_val, ts - 1);
    //
    //                 if res == 1. {
    //                     //genera spike
    //                     v_spike.push(1.0);
    //                     real_spike_vec.push(Spike::new(ts, layer_i, index));
    //                 } else {
    //                     v_spike.push(0.0);
    //                 }
    //
    //             }
    //             spike_mat = DVector::from_vec(v_spike).transpose();
    //             layer.update_layer_cycle(&real_spike_vec);
    //         }
    //
    //         //println!("output from last layer at ts:{}  is {:?}  ",ts,spike_mat.as_slice());
    //         result_vec.push((ts, spike_mat.as_slice().iter().cloned().collect()));
    //     }
    //     println!("{:?}", result_vec);
    //     result_vec
    // }
}
