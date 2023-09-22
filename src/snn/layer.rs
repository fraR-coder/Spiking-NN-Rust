use std::ops::{Index, IndexMut};
extern crate nalgebra as na;

use crate::Model;
use na::{DMatrix, Vector3};

/// A single layer in the neural network
///
/// This contains all the neurons of the layer, as well as the intra-layer weights and input weights from
/// the previous layer.
#[derive(Clone)]
pub struct Layer<M: Model> {
    /// List of all neurons in this layer
    pub(crate) neurons: Vec<M::Neuron>,
    // Matrix of the input weights. For the first layer, this must be a square diagonal matrix.
    pub(crate) input_weights: DMatrix<f64>,
    /// Square matrix of the intra-layer weights
    pub(crate) intra_weights: DMatrix<f64>
    
}

impl<M: Model> Layer<M> {
    pub fn new(neuron: M::Neuron) -> Layer<M> {
        Layer {
            neurons: Vec::new(),
            input_weights: DMatrix::<f64>::from_element(0, 0, 0.0),
            intra_weights: DMatrix::<f64>::from_element(0, 0, 0.0),
        }
    }

    pub fn num_neurons(&self) -> usize {
        self.neurons.len()
    }

    pub fn get_neuron(&self, neuron: usize) -> Option<&M::Neuron> {
        self.neurons.get(neuron)
    }

    pub fn get_neuron_mut(&mut self, neuron: usize) -> Option<&mut M::Neuron> {
        self.neurons.get_mut(neuron)
    }

    pub fn get_intra_weight(&self, from: usize, to: usize) -> Option<f64> {
        self.intra_weights.get((from, to)).copied()
    }
}
