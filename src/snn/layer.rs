extern crate nalgebra as na;

use crate::Model;
use na::{DMatrix};
use nalgebra::DVector;
use crate::snn::Spike;  //guarda metodi from_fn e from_vec

/// A single layer in the neural network
///
/// This contains all the neurons of the layer, as well as the intra-layer weights and input weights from
/// the previous layer.
#[derive(Clone)]
pub struct Layer<M: Model+Clone+'static> {
    /// List of all neurons in this layer
    pub(crate) neurons: Vec<M::Neuron>,
    // Matrix of the input weights. For the first layer, this must be a square diagonal matrix.
    pub(crate) input_weights: DMatrix<f64>,
    /// Square matrix of the intra-layer weights
    pub(crate) intra_weights: DMatrix<f64>
    
}

impl<M: Model + Clone+'static> Layer<M> {
    pub fn new(_neuron: M::Neuron) -> Layer<M> {
        Layer {
            neurons: Vec::new(),
            input_weights: DMatrix::<f64>::from_element(0, 0, 0.0),
            intra_weights: DMatrix::<f64>::from_element(0, 0, 0.0),
        }
    }

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

    pub fn update_layer(&mut self, layer:& mut Layer<M>, vec_spike: & Vec<Spike>) {
        let mut spike_mat=vec![0.0;layer.num_neurons()];
        //creo il vettore che conterrà 1 in corrispondenza di neuorne che spara e 0 altrimenti
        for spike in vec_spike.iter(){
            let neuron_id=spike.neuron_id;
            spike_mat[neuron_id]=1.0;
        }
        
        let dvec= DVector::from_vec(spike_mat);
        let res=dvec.transpose()*&layer.intra_weights;
        
        let neurons=&mut layer.neurons;

       //why cannot access to fields in Neuron
       

    }
    

   
}
