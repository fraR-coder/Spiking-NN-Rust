use std::borrow::Borrow;
use std::fmt::Error;
use crate::Model;
use crate::snn::layer::Layer;
use nalgebra::DMatrix;
use crate::snn::model::lif::LifNeuron;


#[derive(Clone)]
pub struct NN<M: Model> {
    /// All the sorted layers of the neural network
    layers: Vec<Layer<M>>
}

impl<M: Model> NN<M> {
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
}
