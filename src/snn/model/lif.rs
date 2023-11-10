//! Implementation of the Leaky Integrate and Fire (LIF) model for Spiking Neural Networks

use crate::snn::model::Model;

#[derive(Clone, Debug)]
pub struct LifNeuron {
    /// Rest potential
    pub v_rest: f64,
    /// Reset potential
    pub v_reset: f64,
    /// Threshold potential
    pub v_th: f64,
    /// Membrane's time constant. This is the product of its capacity and resistance
    pub tau: f64,

    pub v_mem: f64,
    pub ts_old: u128,
}
/// A struct used to create a specific configuration, simply reusable for other neurons

#[derive(Clone, Debug)]
pub struct Configuration {
    v_rest: f64,
    v_reset: f64,
    v_threshold: f64,
    tau: f64,
}

// IMPLEMENTATION FOR LIF NEURONS & LIF NEURON CONFIG

impl LifNeuron {
    pub fn new(v_rest: f64, v_reset: f64, v_th: f64, tau: f64) -> LifNeuron {
        LifNeuron {
            // parameters
            v_rest,
            v_reset,
            v_th,
            tau,
            v_mem: 0.0, //inizialmente a 0?
            ts_old: 0,
        }
    }

    pub fn from_conf(nc: &Configuration) -> LifNeuron {
        Self::new(nc.v_rest, nc.v_reset, nc.v_threshold, nc.tau)
    }

    pub fn update_vmem(&mut self,val:f64){
        self.v_mem+=val;
    }

    /// Create a new array of n [LifNeuron] structs, starting from a given Configuration.

    pub fn new_vec(conf: Configuration, n: usize) -> Vec<LifNeuron> {
        let mut res: Vec<LifNeuron> = Vec::with_capacity(n);

        for _i in 0..n {
            res.push(LifNeuron::from_conf(&conf));
        }

        res
    }
}
// Implementazione del trait Model per LifNeuron

impl Configuration {
    /// Create a new Configuration, which can be used to build one or more identical neurons.

    pub fn new(v_rest: f64, v_reset: f64, v_threshold: f64, tau: f64) -> Configuration {
        Configuration {
            v_rest,
            v_reset,
            v_threshold,
            tau,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LeakyIntegrateFire;

impl Model for LeakyIntegrateFire {
    type Neuron = LifNeuron;

    type Config = Configuration;

    /// Update the value of current membrane tension.
    /// When the neuron receives one or more impulses, it computes the new tension of the membrane
    /// 
    /// 
   
    /// 
    /// This neuron receives a spike at time of spike _ts_ from a number of its input synapses.
    /// The overall weighted input value of this spike (i.e. the sum, across every lit up input synapse,
    /// of the weight of that synapse) is provided via the _weighted_input_val_ parameter.
    /// 
    /// The output of this function is 1.0 iff the neuron has generated a new spike at time _ts_, or 0.0 otherwise.
    /// 
    /// 

    fn handle_spike(neuron: &mut LifNeuron, weighted_input_val: f64, ts: u128) -> f64 {
        // This early exit serves as a small optimization
        if weighted_input_val == 0.0 {
            return 0.0;
        }
        //println!("ts: {}, ts_old: {}",ts, neuron.ts_old);
        let delta_t: f64 = (ts - neuron.ts_old) as f64;
        neuron.ts_old = ts;

        // compute the new v_mem value
        neuron.v_mem = neuron.v_rest
            + (neuron.v_mem - neuron.v_rest) * (-delta_t / neuron.tau).exp()
            + weighted_input_val;

        if neuron.v_mem > neuron.v_th {
            neuron.v_mem = neuron.v_reset;
            1.0
        } else {
            0.0
        }
    }
    fn update_v_mem(neuron: &mut LifNeuron, val:f64){
        if neuron.v_mem+val >= 0.0 {
            neuron.v_mem+=val;
        } else {
            neuron.v_mem=0.0;
        }
    }

     

}
