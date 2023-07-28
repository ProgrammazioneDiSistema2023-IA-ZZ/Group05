use std::collections::HashMap;
use std::sync::Arc;

use crate::neuron::{Neuron, NeuronParameters};

///SpikingNeuralNetwork module file

///A SpikingNeuralNetwork struct contains a set of all neurons, organized inside a map,
/// so that each one of them can be easily accessed and managed using its i32 ID. Each neuron
/// contains a list of all other neurons adjacent to itself.
/// Neurons are allocated on the heap by using 'Arc' pointers.
/// NOTE: Arc is used instead of Rc because of the multi-thread context
/// provided by the Tokio async runtime.
pub struct SpikingNeuralNetwork {
    neurons: HashMap<i32, Arc<Neuron>>, //map {neuron_id -> Neuron struct}
    neuron_parameters: NeuronParameters, //electrical parameters for network neurons
}

impl SpikingNeuralNetwork {
    ///constructor: this only creates the SpikingNeuralNetwork struct
    /// without populating it.
    pub fn new(neuron_parameters: NeuronParameters) -> Self {
        SpikingNeuralNetwork {
            neurons: HashMap::<i32, Arc<Neuron>>::new(),
            neuron_parameters,
        }
    }

    pub fn load_from_json(&mut self) {
        todo!()
    }
}
