use rand::distributions::Uniform;
use rand::Rng;
use snn::network::{
    json::{LayerData, NetworkData, NeuronData},
    NeuronModel,
};
use std::{fs, path::Path};
fn main() {
    let (nr_inputs, nr_outputs) = (6, 3);
    let time_step_duration_us = 100.0;

    /*network data struct */
    let mut nd = NetworkData {
        time_step_duration_us,
        nr_inputs,
        nr_outputs,
        model: NeuronModel::LeakyIntegrateAndFire,
        layers: Vec::new(),
    };

    /*layers */
    let mut layer1: Vec<NeuronData> = vec![];
    let mut layer2: Vec<NeuronData> = vec![];
    let mut layer3: Vec<NeuronData> = vec![];

    /*Default electric parameters*/
    let v_th = -55.0;
    let v_rest = -70.0;
    let v_reset = -70.0;
    let tau = 10.0;

    /*Random number generator for inter-layer weights (positive)*/
    let mut rng = rand::thread_rng();
    let min_w = 1.0;
    let max_w = 5.0;
    let distr = Uniform::new_inclusive(min_w, max_w);

    /*Random number generator for intra-layer weights (negative)*/
    let mut rng2 = rand::thread_rng();
    let min_w2 = -3.0;
    let max_w2 = -1.0;
    let distr2 = Uniform::new_inclusive(min_w2, max_w2);

    let nr_neurons_l1 = 5;
    /*Layer 1 */
    for i in 0..nr_neurons_l1 {
        let mut weights: Vec<f64> = vec![];
        for _ in 0..nr_inputs {
            weights.push(rng.sample(distr));
        }
        let mut internal_weights: Vec<f64> = vec![];
        for j in 0..nr_neurons_l1 {
            if i == j {
                internal_weights.push(0.0);
            } else {
                internal_weights.push(rng2.sample(distr2));
            }
        }
        layer1.push(NeuronData {
            weights,
            internal_weights,
            v_th,
            v_rest,
            v_reset,
            tau,
        });
    }

    let nr_neurons_l2 = 8;
    /*Layer 2 */
    for i in 0..nr_neurons_l2 {
        let mut weights: Vec<f64> = vec![];
        for _ in 0..nr_neurons_l1 {
            weights.push(rng.sample(distr));
        }
        let mut internal_weights: Vec<f64> = vec![];
        for j in 0..nr_neurons_l2 {
            if i == j {
                internal_weights.push(0.0);
            } else {
                internal_weights.push(rng2.sample(distr2));
            }
        }
        layer2.push(NeuronData {
            weights,
            internal_weights,
            v_th,
            v_rest,
            v_reset,
            tau,
        });
    }

    let nr_neurons_l3 = nr_outputs;
    /*Layer 3 */
    for i in 0..nr_neurons_l3 {
        let mut weights: Vec<f64> = vec![];
        for _ in 0..nr_neurons_l2 {
            weights.push(rng.sample(distr));
        }
        let mut internal_weights: Vec<f64> = vec![];
        for j in 0..nr_neurons_l3 {
            if i == j {
                internal_weights.push(0.0);
            } else {
                internal_weights.push(rng2.sample(distr2));
            }
        }
        layer3.push(NeuronData {
            weights,
            internal_weights,
            v_th,
            v_rest,
            v_reset,
            tau,
        });
    }

    /*Inserting layers */
    nd.layers.append(&mut vec![
        LayerData { neurons: layer1 },
        LayerData { neurons: layer2 },
        LayerData { neurons: layer3 },
    ]);

    let serialized = serde_json::to_string(&nd).unwrap();

    let _ = fs::write(Path::new("sources\\snn_data.json"), serialized).unwrap();
}
