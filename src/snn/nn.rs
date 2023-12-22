use crate::snn::layer::Layer;
use crate::Model;
use nalgebra::{DMatrix, DVector};
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
    /// use your_crate_name::{NN, YourModelType};
    /// use nalgebra::DMatrix;
    ///
    /// let nn = NN::<YourModelType>::new()
    ///     .layer(
    ///         vec![/* neurons for the first layer */],
    ///         DMatrix::from_element(/* input weights */),
    ///         DMatrix::from_element(/* intra-layer weights */),
    ///     )
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
    pub fn solve_single_vec_spike(&mut self, input: Vec<u128>) {
        println!("Enter solve");
        let mut index_input: usize = 0;
        let duration = match input.last().clone() {
            //why cloning
            Some(value) => value.clone(),
            None => 0,
        };
        // creo tanti canali quanti sono i layer
        let num_layers = self.get_num_layers();
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
        for layer_idx in 0..channel_tx.len() {
            let next_tx = if layer_idx < num_layers - 1 {
                Some(channel_tx[layer_idx + 1].clone())
            } else {
                None
            };
            let rx = channel_rx.remove(0);
            let mut layers = self.layers.clone();
            let thread_name = format!("layer_{}", layer_idx);
            let handle = thread:://Builder::new()
                /*.name(thread_name) // Imposta il nome del thread*/
                spawn(move || {
                // il vettore di spike da passare al thread successivo
                let mut layer_output: Vec<Spike> = vec![]; // Output del layer (Spike generati)

                for ts in layer_idx as u128..duration + layer_idx as u128 {
                    // Ricezione degli spike dal layer precedente
                    let input_spike = match rx.recv() {
                        Ok(input_spike) => input_spike,
                        Err(_err) => {
                            println!("Error receiving the vector of spikes, layer: {}", layer_idx);
                            vec![]
                        }
                    };
                    // eseguo i calcoli per aggiornare le tensioni di membrana e riempio il layer_output di spike
                    for neuron_idx in 0..layers[layer_idx].num_neurons() as u128 {
                        let mut sum: f64 = 0.0;
                        let input_spike_tmp = input_spike.clone();
                        for spike in input_spike_tmp.into_iter() {
                            //let neuron=layers[layer_idx].get_neuron(neuron_idx as usize);

                            //sum_whith_injection(neuron_idx,layers[layer_idx],spike);
                            sum += 1.0
                                * layers[layer_idx].input_weights
                                    [[neuron_idx as usize][spike.neuron_id]]; // da reimplementare in una funzione a parte con gli stuck
                        }
                        let res = M::handle_spike(
                            layers[layer_idx]
                                .get_neuron_mut(neuron_idx as usize)
                                .unwrap(),
                            sum,
                            ts,
                        );
                        if res == 1. {
                            layer_output.push(Spike::new(ts, layer_idx, neuron_idx as usize))
                        }
                    }

                    // Invio dei nuovi spike al layer successivo (se presente)
                    if let Some(next_tx) = &next_tx {
                        next_tx
                            .send(layer_output.clone())
                            .expect("Error sending the vector of spikes");
                    } else {
                        for s in layer_output.iter() {
                            println!("Output: {} (thread: {})", s.clone(), thread_name);
                        }
                    }
                    layer_output = vec![];
                } // end for

                // Salvataggio degli spike generati dal layer nell'array
                //let mut layer_data = layer_clone[layer_idx].lock().unwrap();
                //layer_data.extend(layer_output);
            });

            handles.push(handle);
        }
        for ts in 0..duration {
            let spike = match input.get(index_input) {
                Some(value) => value.clone(),
                None => 0,
            };
            index_input += 1;
            for _ in ts..spike {
                first_tx
                    .send(vec![])
                    .expect("Error sending the vector of spikes");
            }
            let mut vec_of_spikes = vec![];
            for _ in 0..self.layers[0].num_neurons() {
                vec_of_spikes.push(Spike::new(spike, 0, 0));
            }
            first_tx
                .send(vec_of_spikes)
                .expect("Error sending the vector of spikes");
        }

        // Attendo il completamento di tutti i thread
        for handle in handles {
            handle.join().expect("kk"); //.expect("Failed to join a thread");
        }
    }

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
    /// use your_crate_name::{NN, YourModelType};
    /// use nalgebra::DMatrix;
    /// use std::sync::{Arc, Mutex};
    ///
    /// let mut nn = NN::<YourModelType>::new();
    /// // Add layers to the neural network
    /// // ...
    ///
    /// let input = vec![(0, vec![1, 2, 3]), (1, vec![4, 5, 6])];
    /// let duration = 10;
    ///
    /// let shared_output = nn.solve_multiple_vec_spike(input, duration);
    ///
    /// // Access the results using the shared_output Arc.
    pub fn solve_multiple_vec_spike(
        &mut self,
        input: Vec<(u128, Vec<u128>)>,
        duration: usize,
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
                let mut counter: u128 = 0;
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
                        let mut sum: f64 = 0.0;
                        let input_spike_tmp = input_spike.clone();
                        let mut s = 0;
                        if !input_spike_tmp.is_empty() {
                            s = input_spike_tmp.clone().get(0).unwrap().ts;
                        }

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
                            layers[layer_idx].update_layer_ciclo(&layer_output);
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
                            counter += 1;
                            neuron_counters[s.neuron_id] += 1;
                        }
                    }
                    layer_output = vec![];
                    ts += 1;
                } // end while

                // Salvataggio degli spike generati dal layer nell'array
                //let mut layer_data = layer_clone[layer_idx].lock().unwrap();
                //layer_data.extend(layer_output);
            });
            handles.push(handle);
        }
        let input_spikes = Spike::vec_of_all_spikes(input);
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
    /// use your_crate_name::{NN, YourModelType, Spike, Layer};
    /// use nalgebra::DMatrix;
    ///
    /// let mut nn = NN::<YourModelType>::new();
    /// // Add layers to the neural network
    /// // ...
    ///
    /// let layer = Layer::new(/* ... */);
    /// let neuron_idx = 0;
    /// let input_spike_tmp = vec![Spike::new(1, 0, 0), Spike::new(2, 1, 0)];
    ///
    /// let sum = nn.calculate_sum(input_spike_tmp, &layer, neuron_idx);
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

  

    pub fn solve_single_thread(mut self, input: Vec<u128>) -> Vec<(u128, Vec<f64>)> {
        let n_neurons_0 = self.layers[0].num_neurons();
        let mut result_vec: Vec<(u128, Vec<f64>)> = Vec::new();

        for ts_i in input {
            let mut ts = ts_i;
            //matrice 1xn
            let mut spike_mat = DVector::from_element(n_neurons_0, 1.0).transpose();
            for (layer_i, layer) in self.layers.iter_mut().enumerate() {
                let mat = &layer.input_weights; //matrice nxn colonna0
                let weighted_matrix = &spike_mat * mat; //matrice 1xn colonna 0 neruone0 ecc..

                ts += 1;
                let mut real_spike_vec = Vec::new();

                let mut v_spike = Vec::new();
                for (index, neuron) in layer.neurons.iter_mut().enumerate() {
                    let weighted_input_val = weighted_matrix[index];
                    println!(
                        "layer: {} neuron:{} ts:{} val:{}",
                        layer_i, index, ts, weighted_input_val
                    );

                    let res = M::handle_spike(neuron, weighted_input_val, ts - 1);

                    if res == 1. {
                        //genera spike
                        v_spike.push(1.0);
                        real_spike_vec.push(Spike::new(ts, layer_i, index));
                    } else {
                        v_spike.push(0.0);
                    }

                    //se genera spike aggiungi al vec spike, segna anche se non ha fatto spike
                }
                spike_mat = DVector::from_vec(v_spike).transpose();
                layer.update_layer_ciclo(&real_spike_vec);
            }

            //alla fine spike vec è vettore che dice quali enuroni del layer hanno fatto spike
            //println!("output from last layer at ts:{}  is {:?}  ",ts,spike_mat.as_slice());
            result_vec.push((ts, spike_mat.as_slice().iter().cloned().collect()));
        }
        println!("{:?}", result_vec);
        result_vec
    }
}
