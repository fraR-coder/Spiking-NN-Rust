extern crate nalgebra as na;

use std::ops::{Deref, DerefMut};
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
    pub fn update_layer(&mut self, vec_spike: & Vec<Spike>) {
        let mut vec:Vec<u128> = vec_spike.iter().map(|s| s.neuron_id as u128).collect();
        let mut vec2:Vec<f64> = vec![];
        let mut i:u128 = 0;
        while i<self.num_neurons() as u128{
            for _ in i .. if vec.len()>0 {vec.remove(0)} else { self.num_neurons() as u128 }{
                vec2.push(0.0);
                i+=1;
            }
            if vec2.len()<self.num_neurons(){
                vec2.push(1.0);
            }
            i+=1;
        }
        let d_vec= DVector::from_vec(vec2);
        let res=d_vec.transpose() * &self.intra_weights;
        for (neuron_idx, weight) in res.iter().enumerate(){
            if neuron_idx<self.num_neurons(){
                M::update_v_mem(self.get_neuron_mut(neuron_idx).unwrap(),*weight);
            }
        }
    }


    pub fn update_layer_ciclo(&mut self, vec_spike: & Vec<Spike>) {
        let mut vec:Vec<u128> = vec_spike.iter().map(|s| s.neuron_id as u128).collect();
        let mut vec2:Vec<f64> = vec![];
        let mut i:u128 = 0;
       
       

            for neuron_idx in 0.. self.num_neurons(){

                let mut sum=0.;
                for spike in vec_spike.iter(){
                    
                    sum += 1.0*self.intra_weights[(spike.neuron_id, neuron_idx as usize)]; // da reimplementare in una funzione a parte con gli stuck


                }
                M::update_v_mem(self.get_neuron_mut(neuron_idx).unwrap(),sum);

            }
        }

}