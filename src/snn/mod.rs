use std::fmt;

pub mod layer;
pub mod model;
pub mod nn;
pub mod resilience;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
pub struct Spike {
    /// timestamp of when the spike occurs
    pub ts: u128,
    /// Index of the layer, the neuron of which generated the spike
    pub layer_id: usize,
    /// Index of the neuron which generated the spike
    pub neuron_id: usize,
}
impl Spike {
    // Create a new spike at time `ts` for neuron `neuron_id`
    pub fn new(ts: u128, layer_id: usize, neuron_id: usize) -> Spike {
        Spike { ts, layer_id, neuron_id }
    }
    // lo trovo inutile
    /*
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
    */

    //recive a matrix where each line is vector of Spikes and merge all the Spikes in a terminal verctor
    pub fn vec_of_all_spikes(spikes: Vec<(u128,Vec<u128>)>) -> Vec<Spike> {
        let mut res: Vec<Spike> = spikes.into_iter().flat_map(|(neuron_id, spikes_vector)| {
            spikes_vector.into_iter().map(move |ts| Spike {
                ts,
                layer_id: 0,
                neuron_id: neuron_id as usize
            })
        }).collect();

        res.sort(); // ascending order by default because of the `Ord` trait
        res
    }
}

impl fmt::Display for Spike {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Spike(ts: {}, layer_id: {}, neuron_id: {})", self.ts, self.layer_id, self.neuron_id)
    }
}



