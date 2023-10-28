use crate::snn::model::lif::*;
use crate::NN;
use rand::Rng;
use std::sync::{Arc, Mutex};

/// The struct that contains the input configuration
#[derive(Clone)]
pub struct Resilience {
    /// List of all components
    pub(crate) components: Vec<String>,
    /// Stuck type: 0 for stuck_at_0, 1 for stuck_at_1, 2 for transient_bit.
    pub(crate) stuck_type: u8,
    /// The number of times to do the resilience test
    pub(crate) times: u128,
}

impl Resilience {
    pub fn new(components: Vec<String>, stuck_type: u8, times: u128) -> Resilience {
        return Resilience {
            components,
            stuck_type,
            times,
        };
    }

    pub fn get_rand_component(&self) -> String {
        return self.components[rand::thread_rng().gen_range(0..self.components.len())].clone();
    }

    pub fn execute_resilience_test(
        &self,
        mut snn: NN<LeakyIntegrateFire>,
        input: Vec<(u128, Vec<u128>)>,
    ) {
        let mut stuck: bool = true;
        let mut count_right_outputs: u64 = 0;
        let right_output = snn.clone().solve_multiple_vec_spike(input.clone(), 11);
        println!("{:?}", right_output);

        match self.stuck_type {
            0 => stuck = false,
            1 => stuck = true,
            _ => {
                println!("Error: wrong stuck type")
            }
        }
        for _ in 0..self.times {
            let mut snn_tmp = snn.clone();
            //println!(solution);
            //println!("Type: {}",&self.get_rand_component().to_lowercase() as &str);
            let component = self.get_rand_component().to_lowercase();

            match &component as &str {
                ("neurons" | "n" | "neu" | "neuron") => {
                    let rand_layer_idx = rand::thread_rng().gen_range(0..snn_tmp.get_num_layers());
                    let rand_neuron_idx = rand::thread_rng()
                        .gen_range(0..snn_tmp.layers[rand_layer_idx].num_neurons());
                    snn_tmp.layers[rand_layer_idx].stuck_bit_neuron(
                        stuck,
                        rand_neuron_idx,
                        vec!["v_rest", "v_reset", "v_tau", "v_th"]
                            .get(rand::thread_rng().gen_range(0..4))
                            .unwrap()
                            .to_string(),
                    );
                }
                _ => {
                    println!("Error unknown component");
                }
            }
            let res = snn_tmp.solve_multiple_vec_spike(input.clone(), 11);
            //println!("{:?}", res);
            if are_equal(&res, &right_output) {
                count_right_outputs += 1;
            }
        }
        println!(
            "for this SNN, running the stuck bit {} times, in {}% of cases the output is the same",
            self.times,
            count_right_outputs as f64 / self.times as f64 * 100.0
        );
    }
}
pub fn are_equal(
    a: &Arc<Mutex<Vec<(u128, Vec<u128>)>>>,
    b: &Arc<Mutex<Vec<(u128, Vec<u128>)>>>,
) -> bool {
    // Lock entrambi i Mutex.
    let a_guard = a.lock().unwrap();
    let b_guard = b.lock().unwrap();
    *a_guard == *b_guard
}
