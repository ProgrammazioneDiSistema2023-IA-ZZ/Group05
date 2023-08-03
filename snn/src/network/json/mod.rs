use crate::network;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize)]
pub struct NetworkData {
    pub time_step_duration_us: f64,
    pub nr_inputs: usize,
    pub nr_outputs: usize,
    pub layers: Vec<LayerData>,
}

#[derive(Serialize, Deserialize)]
pub struct NeuronData {
    pub weights: Vec<f64>,
    pub internal_weights: Vec<f64>,
    pub v_th: f64,
    pub v_rest: f64,
    pub v_reset: f64,
    pub tau: f64,
}

#[derive(Serialize, Deserialize)]
pub struct LayerData {
    pub neurons: Vec<NeuronData>,
}

pub fn load_from_file(path: &str) -> network::Network {
    let json_str = fs::read_to_string(path).expect("Couldn't read file");
    let nd: NetworkData = serde_json::from_str(&json_str).expect("Incorrect file format");

    let mut network = network::Network::new(nd.time_step_duration_us, nd.nr_inputs, nd.nr_outputs);

    for layer_data in nd.layers {
        let mut layer = Vec::<network::neuron::Neuron>::new();
        for neuron_data in layer_data.neurons {
            let mut neuron = network::neuron::Neuron::new(
                neuron_data.v_th,
                neuron_data.v_rest,
                neuron_data.v_reset,
                neuron_data.tau,
            );
            neuron.set_weights(neuron_data.weights);
            neuron.set_internal_weights(neuron_data.internal_weights);
            layer.push(neuron);
        }
        network.add_layer(layer);
    }

    return network;
}
