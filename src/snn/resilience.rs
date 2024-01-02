//! # Spiking Neural Network Resilience Library
//!
//! This Rust library extends the spiking neural network (SNN) framework to incorporate resilience mechanisms.
//! It introduces a `Resilience` struct to configure and execute resilience tests on SNNs, evaluating their performance under faulty conditions.
//!
//! ## Overview
//!
//! The library builds upon the Spiking Neural Network Library, providing additional functionality for resilience testing.
//! The `Resilience` struct allows users to specify components, stuck types, and the number of resilience test repetitions.
//! Resilience tests are executed by introducing faults in randomly selected components of the SNN and observing the output variations.
//!
//! ## Usage
//!
//! The resilience library includes the following main components:
//!
//! - **Resilience Structure (`Resilience`):** Represents the configuration for a resilience test.
//! - **Stuck Enum (`Stuck`):** Defines different stuck types, such as stuck-at-0, stuck-at-1, and transient bit.
//! - **Utility Function (`are_equal`):** Compares two sets of spikes for equality.
//!
//! ## Resilience Testing
//!
//! To perform resilience testing, users can create an instance of the `Resilience` struct and execute the test on a given SNN.
//! The library supports various fault types, including faults in neurons, membrane potentials, and specific components like full adders and comparators.
//!
//! ```rust
//! use spiking_nn_resilience::{NN, LeakyIntegrateFire, Resilience, are_equal};
//! use spiking_nn_resilience::model::Stuck;
//!
//! // Assuming 'snn' is an instance of the NN structure
//!
//! let resilience_config = Resilience::new(
//!     /* components */ vec!["neurons".to_string(), "vmem".to_string(), "fullAdder".to_string()],
//!     /* stuck_type */ Stuck::StuckAt1,
//!     /* times */ 100,
//! );
//!
//! resilience_config.execute_resilience_test(snn.clone(), /* input spikes */ vec![(0, vec![1, 2, 3]), (1, vec![2, 4, 6])]);
//! ```
//!
//! ## Utility Function
//!
//! The `are_equal` function compares two sets of spikes for equality. It is used to check if the output spikes from the resilience tests match the expected output.
//!
//! ```rust
//! // Assuming 'output_a' and 'output_b' are instances of Arc<Mutex<Vec<(u128, Vec<u128>)>>>
//!
//! let are_outputs_equal = are_equal(&output_a, &output_b);
//!
//! if are_outputs_equal {
//!     println!("Output spikes are equal.");
//! } else {
//!     println!("Output spikes differ.");
//! }
//! ```

use crate::snn::model::lif::*;
use crate::NN;
use rand::Rng;
use std::sync::{Arc, Mutex};
use crate::snn::Spike;

use super::model::Stuck;

/// The struct that contains the input configuration
#[derive(Clone)]
pub struct Resilience {
    /// List of all components
    pub(crate) components: Vec<String>,
    /// Stuck type: 0 for stuck_at_0, 1 for stuck_at_1, 2 for transient_bit.
    pub(crate) stuck_type: Stuck,
    /// The number of times to do the resilience test
    pub(crate) times: u128,
}

impl Resilience {
    /// Creates a new `Resilience` instance with the given components, stuck type, and test repetition count.
    ///
    /// # Arguments
    ///
    /// * `components` - List of all components to be tested for resilience.
    /// * `stuck_type` - Stuck type: 0 for stuck_at_0, 1 for stuck_at_1, 2 for transient_bit.
    /// * `times` - The number of times to perform the resilience test.
    ///
    /// # Returns
    ///
    /// * `Resilience` - A new `Resilience` instance.
    pub fn new(components: Vec<String>, stuck_type: Stuck, times: u128) -> Resilience {
        return Resilience {
            components,
            stuck_type,
            times,
        };
    }

    pub fn get_rand_component(&self) -> String {
        return self.components[rand::thread_rng().gen_range(0..self.components.len())].clone();
    }

    /// Creates a new `Resilience` instance with the given components, stuck type, and test repetition count.
    ///
    /// # Arguments
    ///
    /// * `components` - List of all components to be tested for resilience.
    /// * `stuck_type` - Stuck type: 0 for stuck_at_0, 1 for stuck_at_1, 2 for transient_bit.
    /// * `times` - The number of times to perform the resilience test.
    ///
    /// # Returns
    ///
    /// * `Resilience` - A new `Resilience` instance.
    pub fn execute_resilience_test(
        &self,
        snn: NN<LeakyIntegrateFire>,
        input: Vec<(u128, Vec<u128>)>,
    ) -> Arc<Mutex<Vec<(u128, Vec<u128>)>>> {

        let mut count_right_outputs: u64 = 0;
        let time_init = std::time::Instant::now();
        let right_output = snn.clone().solve_multiple_vec_spike(input.clone());
        // println!("{:?}", right_output);
        println!("Executing resilience test for given Spiking Neural Network. Total number of input spikes: {:?}", Spike::vec_of_all_spikes(input.clone()));
        println!("Total number of input spikes: {}", Spike::vec_of_all_spikes(input.clone()).len());
        println!("Total number of output spikes: {}", Spike::vec_of_all_spikes(right_output.lock().unwrap().clone()).len());
        for _ in 0..self.times {
            let mut snn_tmp = snn.clone();
            //println!(solution);
            //println!("Type: {}",&self.get_rand_component().to_lowercase() as &str);

            //select a random component between the one chosen by the user
            let component = self.get_rand_component().to_lowercase();

            match &component as &str {
                "neurons" | "n" | "neu" | "neuron" | "neuroni" | "neurone" => {
                    // println!("chose neuron");
                    let rand_layer_idx = rand::thread_rng().gen_range(0..snn_tmp.get_num_layers());
                    let rand_neuron_idx = rand::thread_rng()
                        .gen_range(0..snn_tmp.layers[rand_layer_idx].num_neurons());
                    snn_tmp.layers[rand_layer_idx].stuck_bit_neuron(
                        self.stuck_type.clone(),
                        rand_neuron_idx,
                        vec!["v_rest", "v_reset", "v_tau", "v_th"]
                            .get(rand::thread_rng().gen_range(0..4))
                            .unwrap()
                            .to_string(),
                    );
                }
                "vmem" | "potenziale di membrana" | "membrane potential" | "membrane" | "membrana" | "v_mem"=> {
                    // println!("chose vmem");
                    let rand_layer_idx = rand::thread_rng().gen_range(0..snn_tmp.get_num_layers());
                    let rand_neuron_idx = rand::thread_rng()
                        .gen_range(0..snn_tmp.layers[rand_layer_idx].num_neurons());
                    snn_tmp.layers[rand_layer_idx].stuck_bit_neuron(
                        self.stuck_type.clone(),
                        rand_neuron_idx,
                        "v_mem".to_string(),
                    )
                }

                "fullAdder" | "full adder" | "full-adder" | "full_adder" | "adder" | "sommatore" => {
                    // println!("chose full adder");
                    let rand_layer_idx = rand::thread_rng().gen_range(0..snn_tmp.get_num_layers());
                    let rand_neuron_idx = rand::thread_rng()
                        .gen_range(0..snn_tmp.layers[rand_layer_idx].num_neurons());
                    snn_tmp.layers[rand_layer_idx].stuck_bit_neuron(
                        self.stuck_type.clone(),
                        rand_neuron_idx,
                        "full adder".to_string(),
                    )
                }
                "comparatore" | "comparator" | "threshold" | "threashold comparator" => {
                    // println!("chose comparator");
                    let rand_layer_idx = rand::thread_rng().gen_range(0..snn_tmp.get_num_layers());
                    let rand_neuron_idx = rand::thread_rng()
                        .gen_range(0..snn_tmp.layers[rand_layer_idx].num_neurons());
                    snn_tmp.layers[rand_layer_idx].stuck_bit_neuron(
                        self.stuck_type.clone(),
                        rand_neuron_idx,
                        "comparator".to_string(),
                    )
                }
                _ => {
                    println!("Error unknown component");
                }
            }
            let res = snn_tmp.solve_multiple_vec_spike(input.clone());
            //println!("{:?}", res);
            if are_equal(&res, &right_output) {
                count_right_outputs += 1;
            }
        }

        println!(
            "For this SNN, running the stuck bit \"{:?}\" on {} for {} times, in {}% of cases the output is the same.\nThe process took {:?} to be finished.",
            self.stuck_type,
            self.components.join(", "),
            self.times,
            count_right_outputs as f64 / self.times as f64 * 100.0,
            std::time::Instant::now()-time_init
        );
        return right_output;
    }
}
pub fn are_equal(
    a: &Arc<Mutex<Vec<(u128, Vec<u128>)>>>,
    b: &Arc<Mutex<Vec<(u128, Vec<u128>)>>>,
) -> bool {
    // Lock both Mutex.
    let a_guard = a.lock().unwrap();
    let b_guard = b.lock().unwrap();
    *a_guard == *b_guard
}
