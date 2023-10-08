use crate::Model;
use crate::snn::layer::Layer;
use nalgebra::DMatrix;
use std::sync::{mpsc};
use std::thread;

use super::Spike;


#[derive(Clone)]
pub struct NN<M: Model+Clone+'static> {
    /// All the sorted layers of the neural network
    pub layers: Vec<Layer<M>>
}

impl<M: Model+Clone> NN<M> {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
        }
    }
    pub fn layer(
        mut self,
        neurons: Vec<M::Neuron>,
        input_weights: DMatrix<f64>,
        intra_weights: DMatrix<f64>,
    ) -> Result<Self, String>
    {
        let len_last_layer = self.layers.last().map(|l| l.neurons.len()).unwrap_or(0);
        let n = neurons.len();

        // Check layer len not zero
        if n == 0 {
            return Err("The number of neurons should be at least 1".to_string());
        }

        // Check size compatibilities
        if intra_weights.len() != n*n {
            return Err("Incompatible intra weight matrix".to_string());
        }

        if input_weights.len() != (
            if len_last_layer == 0 { n*n } else { len_last_layer * n }
        ) {
            return Err("Incompatible intra weight matrix".to_string());
        }

        // Finally, insert layer into nn
        let new_layer = Layer {
            neurons,
            input_weights,
            intra_weights
        };
        self.layers.push(new_layer);

        Ok(self)
    }
    pub fn get_num_layers(&self) -> usize{
        return self.layers.len()
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
    pub fn solve(&mut self, input: Vec<u128>) {
        let index_input: usize=0;
        let duration =  match input.last().clone() {
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
            if i==0{
                first_tx=tx.clone();
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
            //let layer_clone = self.layers.clone(); // Cloniamo i dati del layer per passarli al thread
            let mut layers = self.layers.clone();
            let thread_name=format!("layer_{}", layer_idx);
            let handle = thread::Builder::new()
                .name(thread_name) // Imposta il nome del thread
                .spawn(move || {
                // il vettore di spike da passare al thread successivo
                let mut layer_output: Vec<Spike> = vec![]; // Output del layer (Spike generati)

                for ts in layer_idx as u128..duration+layer_idx as u128 {
                    // Ricezione degli spike dal layer precedente
                    let input_spike = match rx.recv() {
                        Ok(input_spike) => {
                            input_spike
                        }
                        Err(_err) => {
                            println!("Error receiving the vector of spikes, layer: {}",layer_idx);
                            vec![]
                        }
                    };
                    // eseguo i calcoli per aggiornare le tensioni di membrana e riempio il layer_output di spike
                    for neuron_idx in 0..layers[layer_idx].num_neurons() as u128{
                        let mut sum:f64 = 0.0;
                        let input_spike_tmp = input_spike.clone();
                        for spike in input_spike_tmp.into_iter(){
                            sum += 1.0*layers[layer_idx].input_weights[[neuron_idx as usize][spike.neuron_id]]; // da reimplementare in una funzione a parte con gli stuck
                        }
                        let res = M::handle_spike(layers[layer_idx].get_neuron_mut(neuron_idx as usize).unwrap(),sum, ts);
                        if res==1. {
                            layer_output.push(Spike::new(ts,layer_idx,neuron_idx as usize))
                        }
                        layer_output=vec![]
                    }

                    // Invio dei nuovi spike al layer successivo (se presente)
                    if let Some(next_tx) = &next_tx {
                        println!("Thread: {}",thread::current().name().unwrap_or("Unnamed"));
                        next_tx.send(layer_output.clone()).expect("Error sending the vector of spikes");
                    } else {
                        for s in layer_output.iter(){
                            println!("Output: {} (thread: {})",s.clone(),thread::current().name().unwrap_or("Unnamed"));
                        }
                    }
                } // end for

                // Salvataggio degli spike generati dal layer nell'array
                //let mut layer_data = layer_clone[layer_idx].lock().unwrap();
                //layer_data.extend(layer_output);
            });

            handles.push(handle);
        }
        for ts in 0..duration{
            let spike = match input.get(index_input){
                Some(value) => value.clone(),
                None => 0,
            };
            for _ in ts..spike{
                first_tx.send(vec![]).expect("Error sending the vector of spikes");
            }
            first_tx.send(vec![Spike::new(spike,0,0)]).expect("Error sending the vector of spikes");
        }

        // Attendo il completamento di tutti i thread
        for handle in handles {
            handle.expect("Failed to join a thread");
        }
    }

    /*
    pub fn solve(&mut self, input: Vec<Vec<u8>>, duration: u128) {
        for t in 0..duration {
            // Simulate each time step
            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                let mut spike_vec: Vec<Spike> = Vec::new();

                for (neuron_idx, neuron) in layer.neurons.iter_mut().enumerate() {
                    let weighted_input = calculate_weighted_input(layer_idx, neuron_idx, &self.layers, &input, t);
                    let spike = neuron.handle_spike(weighted_input, t);

                    if spike == 1.0 {
                        spike_vec.push(Spike::new(t, neuron_idx));
                    }
                }

                // Handle spike propagation to the next layer (if not the last layer)
                if layer_idx < self.layers.len() - 1 {
                    let next_layer = &mut self.layers[layer_idx + 1];
                    for (neuron_idx, spike) in spike_vec.iter().enumerate() {
                        if let Some(weight) = next_layer.get_intra_weight_mut(neuron_idx, neuron_idx) {
                            // Propagate spike with weight
                            let spike = Spike::new(t, neuron_idx);
                            if let Some(next_neuron) = next_layer.get_neuron_mut(neuron_idx) {
                                next_neuron.handle_spike(*weight * spike, t);
                            }
                        }
                    }
                }
            }
        }
        fn calculate_weighted_input<M: Model>(
            layer_idx: usize,
            neuron_idx: usize,
            layers: &Vec<Layer<M>>,
            input: &Vec<Vec<u8>>,
            t: u128,
        ) -> f64 {
            let mut weighted_input = 0.0;

            // Calcola l'input pesato dai neuroni di input
            if layer_idx == 0 {
                for (input_neuron_idx, input_spike) in input[layer_idx].iter().enumerate() {
                    // Supponiamo che il peso sia 1.0 per semplicità
                    weighted_input += *input_spike as f64;
                }
            } else {
                // Calcola l'input pesato dai neuroni nei livelli precedenti
                for prev_neuron_idx in 0..layers[layer_idx - 1].num_neurons() {
                    if let Some(weight) = layers[layer_idx - 1].get_intra_weight(prev_neuron_idx, neuron_idx) {
                        // Supponiamo che ci sia una connessione e calcoliamo l'input pesato
                        if let Some(prev_neuron) = layers[layer_idx - 1].get_neuron(prev_neuron_idx) {
                            // Supponiamo che prev_neuron.handle_spike() restituisca lo spike emesso
                            let prev_neuron_spike = prev_neuron.handle_spike(0.0, t);
                            weighted_input += prev_neuron_spike * weight;
                        }
                    }
                }
            }

            weighted_input
        }
*/


//NON SI PUO FARE RECURSIVE, e dovrei fare visita in ampiezza non profondità :(
//I LAYER NON SANNO CHI è IL LAYER SUCCESSIVO
    /*
    pub fn solve(mut self,input:Vec<u8>){


        for _i in input{
            let mut spike_vec:Vec<Spike>=Vec::new();
            for layer in self.layers.iter(){
                
                for lif in layer.neurons.iter(){

                    //calcola valore input pesato con dot product, usa spike vec 
                    //chiama handle spike  (è difficile senno l'avrei fatto ora )
                    //se genera spike aggiungi al vec spike, segna anche se non ha fatto spike

                    
                }
            }

            //alla fine spike vec è vettore che dice quali enuroni del layer hanno fatto spike

        }

    }*/

}
