use rand::Rng;
use crate::NN;
use crate::snn::model::{lif::*, Model};

/// The struct that contains the input configuration
#[derive(Clone)]
pub struct Resilience<> {
    /// List of all components
    pub(crate) components: Vec<String>,
    /// Stuck type: 0 for stuck_at_0, 1 for stuck_at_1, 2 for transient_bit.
    pub(crate) stuck_type: u8,
    /// The number of times to do the resilience test
    pub(crate) times: u128,

}


impl Resilience<> {
    pub fn new(components: Vec<String>, stuck_type: u8, times: u128) -> Resilience<> {
        return Resilience{
            components,
            stuck_type,
            times
        }
    }

    pub fn get_rand_component(&self) -> String{
        return self.components[rand::thread_rng().gen_range(0..self.components.len() as usize)].clone();
    }

    pub fn execute_resilience_test(&self, mut snn: NN<LeakyIntegrateFire>, input: Vec<(u128, Vec<u128>)>) {
        let mut snn2 = snn.clone();
        snn.solve_multiple_vec_spike(input.clone(), 11);
        let mut stuck:bool=true;
        match self.stuck_type {
            0 => {stuck=false}
            1 => {stuck=true}
            _ => {println!("Error: wrong stuck type")}
        }
        //println!(solution);
        println!("Type: {}",&self.get_rand_component().to_lowercase() as &str);
        let component = self.get_rand_component().to_lowercase();

        match &component as &str {
            ("neurons" | "n" | "neu" | "neuron") => {
                   let rand_layer_idx = rand::thread_rng().gen_range(0..snn2.get_num_layers());
                   let rand_neuron_idx = rand::thread_rng().gen_range(0..snn2.layers[rand_layer_idx].num_neurons());
                   let neuron = snn2
                       .layers[rand_layer_idx]
                       .get_neuron_mut(rand_neuron_idx).unwrap();

                   snn2.layers[3].stuck_bit_neuron(true, 3,"v_rest".to_string()  );
               }
            _ => {
                println!("Error unknown component");
            }
        }
        snn2.solve_multiple_vec_spike(input,11);
    }
}