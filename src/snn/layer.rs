//! `Layer` type for each layer of the neural network

use std::ops::{Index, IndexMut};
use ndarray::Array2;
use crate::Model;

/// A single layer in the neural network
/// 
/// This contains all the neurons of the layer, as well as the intra-layer weights and input weights from
/// the previous layer.
#[derive(Clone)]
pub struct Layer<M: Model> {
    /// List of all neurons in this layer
    pub(crate) neurons: Vec<M::Neuron>,
    /// Matrix of the input weights. For the first layer, this must be a square diagonal matrix.
    pub(crate) input_weights: Array2<f64>,
    /// Square matrix of the intra-layer weights
    pub(crate) intra_weights: Array2<f64>
}