use std::fmt;

pub mod layer;
pub mod model;
pub mod nn;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Spike {
    /// timestamp of when the spike occurs
    pub ts: u128,
    /// Index of the neuron this spike applies to inside its layer
    pub neuron_id: usize,
}

impl Spike {
    // Create a new spike at time `ts` for neuron `neuron_id`
    pub fn new(ts: u128, neuron_id: usize) -> Spike {
        Spike { ts, neuron_id }
    }

    // Create an array of spikes for a single neuron, given its ID.
    pub fn vec_of_spike_for(neuron_id: usize, ts_vec: Vec<u128>) -> Vec<Spike> {
        let mut spike_vec: Vec<Spike> = Vec::with_capacity(ts_vec.len());

        //Creating the Spikes array for a single Neuron
        for ts in ts_vec.into_iter() {
            spike_vec.push(Spike::new(ts, neuron_id));
        }

        //Order the ts vector
        spike_vec.sort();

        spike_vec
    }

    //recive a matrix where each line is vector of Spikes and merge all the Spikes in a terminal verctor
    pub fn vec_of_all_spikes(spikes: Vec<Vec<Spike>>) -> Vec<Spike> {
        let mut res: Vec<Spike> = Vec::new();

        res=spikes.into_iter().flatten().collect();

        res.sort(); //ascending
    
        res
    }
}

impl fmt::Display for Spike {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Spike(ts: {}, neuron_id: {})", self.ts, self.neuron_id)
    }
}