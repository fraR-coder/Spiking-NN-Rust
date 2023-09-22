
#[derive(Clone)]
pub struct NNBuilder<M: Model> {
    /// Inner, growing [NN]
    nn: NN<M>,
    
}

impl<M: Model> NNBuilder<M> {
    /// Create a new dynamically sized instance of [NNBuilder].
   
    pub fn new() -> Self {
        Self { nn: Self::new_nn() }
    }

     /// Create a new, empty [NN]
     fn new_nn() -> NN<M> {
        NN {
            layers: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct NN<M: Model> {
    /// All the sorted layers of the neural network
    layers: Vec<Layer<M>>
}

impl<M: Model> NN<M> {

}
