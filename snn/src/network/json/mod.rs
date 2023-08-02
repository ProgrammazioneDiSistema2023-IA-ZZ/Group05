use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NetworkData {
    pub nr_inputs: usize,
    pub nr_outputs: usize,
    pub layers: Vec<LayerData>,
}

#[derive(Serialize, Deserialize)]
pub struct NeuronData {
    pub weights: Vec<f64>,
    pub internal_weights: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct LayerData {
    pub neurons: Vec<NeuronData>,
}
