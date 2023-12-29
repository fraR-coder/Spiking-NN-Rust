extern crate nalgebra as na;

use crate::snn::Spike;
use crate::Model;
use na::DMatrix;
use nalgebra::DVector;

use super::model::lif::LeakyIntegrateFire;
use super::model::Stuck;
use super::nn::NN; //guarda metodi from_fn e from_vec

/// A single layer in the neural network
///
/// This contains all the neurons of the layer, as well as the intra-layer weights and input weights from
/// the previous layer.
#[derive(Clone)]
pub struct Layer<M: Model + Clone + 'static> {
    /// List of all neurons in this layer
    pub(crate) neurons: Vec<M::Neuron>,
    // Matrix of the input weights. For the first layer, this must be a square diagonal matrix.
    pub(crate) input_weights: DMatrix<f64>,
    /// Square matrix of the intra-layer weights
    pub(crate) intra_weights: DMatrix<f64>,
}

impl<M: Model + Clone + 'static> Layer<M> {
    // pub fn new(_neuron: M::Neuron) -> Layer<M> {
    //     Layer {
    //         neurons: Vec::new(),
    //         input_weights: DMatrix::<f64>::from_element(0, 0, 0.0),
    //         intra_weights: DMatrix::<f64>::from_element(0, 0, 0.0),
    //     }
    // }

    /// Return the number of neurons in this [Layer]
    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    /// Get the specified neuron, or [None] if the index is out of bounds.
    pub fn get_neuron(&self, neuron: usize) -> Option<&M::Neuron> {
        self.neurons.get(neuron)
    }

    /// Get a mutable reference to the specified neuron, or [None] if the index is out of bounds.
    pub fn get_neuron_mut(&mut self, neuron: usize) -> Option<&mut M::Neuron> {
        self.neurons.get_mut(neuron)
    }
    /// Get the intra-layer weight from and to the specified neurons, or [None] if any index is out of bounds.
    pub fn get_intra_weight(&self, from: usize, to: usize) -> Option<f64> {
        self.intra_weights.get((from, to)).copied()
    }
    /// Get a mutable reference to the intra-layer weight from and to the specified neurons, or [None] if any index is out of bounds.
    pub fn get_intra_weight_mut(&mut self, from: usize, to: usize) -> Option<&mut f64> {
        self.intra_weights.get_mut((from, to))
    }

    /// Returns an ordered iterator over all the neurons in this layer.
    pub fn iter_neurons(&self) -> <&Vec<M::Neuron> as IntoIterator>::IntoIter {
        self.neurons.iter()
    }

    /// Updates the layer's neuron membrane potentials based on the input spikes.
    ///
    /// # Arguments
    ///
    /// * `vec_spike` - Vector of spikes representing the input to the layer.

    pub fn update_layer(&mut self, vec_spike: &Vec<Spike>) {
        let mut spike_mat = vec![0.0; self.num_neurons()];
        //creo il vettore che conterrà 1 in corrispondenza di neuorne che spara e 0 altrimenti
        for spike in vec_spike.iter() {
            let neuron_id = spike.neuron_id;
            spike_mat[neuron_id] = 1.0;
        }

        let dvec = DVector::from_vec(spike_mat);
        let res = dvec.transpose() * &self.intra_weights;
        for (neuron_idx, weight) in res.iter().enumerate() {
            if neuron_idx < self.num_neurons() {
                M::update_v_mem(self.get_neuron_mut(neuron_idx).unwrap(), *weight);
            }
        }
    }

    pub fn update_layer_ciclo(&mut self, vec_spike: &Vec<Spike>) {
        for neuron_idx in 0..self.num_neurons() {
            let mut sum = self.calculate_sum(vec_spike.clone(), neuron_idx as u128);
            M::update_v_mem(self.get_neuron_mut(neuron_idx).unwrap(), sum);
        }
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
    ///
    /// use nalgebra::DMatrix;
    /// use spiking_nn_resilience::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
    /// use spiking_nn_resilience::{Layer, Model, NN};
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
    /// let neuron_idx = 0;
    /// let input_spike_tmp = vec![Spike::new(1, 0, 0), Spike::new(2, 1, 0)];
    ///
    /// let sum = nn.unwrap().layers.get(0).unwrap().calculate_sum(input_spike_tmp, neuron_idx);
    /// ```
    pub fn calculate_sum(&self, input_spike_tmp: Vec<Spike>, neuron_idx: u128) -> f64 {
        let neuron = self.get_neuron(neuron_idx as usize).unwrap();

        let inputs_to_sum: Vec<f64> = input_spike_tmp
            .into_iter()
            .map(|spike| self.intra_weights[(spike.neuron_id, neuron_idx as usize)])
            .collect();

        if let Some(mut heap_vec) = M::get_heap(neuron) {
            heap_vec.sum_all(&inputs_to_sum)
        } else {
            inputs_to_sum.iter().sum()
        }
    }

    /// Applies a stuck bit to a specific neuron's parameter.
    ///
    /// # Arguments
    ///
    /// * `stuck` - The type of stuck bit to apply.
    /// * `neuron_id` - The index of the neuron to apply the stuck bit.
    /// * `neuron_data` - The specific parameter of the neuron to apply the stuck bit.

    pub fn stuck_bit_neuron(&mut self, stuck: Stuck, neuron_id: usize, neuron_data: String) {
        match neuron_data.as_str() {
            "v_th" => {
                M::update_v_th(self.get_neuron_mut(neuron_id).unwrap(), stuck);
            }
            "v_rest" => {
                //println!("Entrato in v_rest del layer: ");
                M::update_v_rest(self.get_neuron_mut(neuron_id).unwrap(), stuck);
            }
            "v_reset" => {
                M::update_v_reset(self.get_neuron_mut(neuron_id).unwrap(), stuck);
            }
            "v_tau" => {
                M::update_tau(self.get_neuron_mut(neuron_id).unwrap(), stuck);
            }
            "v_mem" => {
                M::use_v_mem_with_injection(self.get_neuron_mut(neuron_id).unwrap(), stuck);
            }
            //logic for full adder
            "full adder" => {
                let inputs: Vec<f64> = self
                    .input_weights
                    .column(neuron_id)
                    .iter()
                    .cloned()
                    .collect();

                println!("number of inputs for neuron is {}", inputs.len());
                M::use_heap(self.get_neuron_mut(neuron_id).unwrap(), stuck, inputs);
            }
            //logic for comparator
            "comparator" => {
                M::use_comparator(self.get_neuron_mut(neuron_id).unwrap(), stuck);
            }

            _ => {
                println!("Error: invalid parameter");
            }
        }
    }
}
