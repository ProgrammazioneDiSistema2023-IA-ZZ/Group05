use snn::network::{neuron::Neuron, Network};
use snn::register::Register;

fn main() {
    let mut l0 = Vec::new();
    let mut l1 = Vec::new();

    for _ in 0..4 {
        let mut neuron = Neuron::default();
        neuron.weights.append(&mut vec![
            Register::new(4.0),
            Register::new(4.0),
            Register::new(4.0),
            Register::new(4.0),
        ]);
        l0.push(neuron);
    }

    for _ in 0..3 {
        let mut neuron = Neuron::default();
        neuron.weights.append(&mut vec![
            Register::new(3.5),
            Register::new(3.5),
            Register::new(3.5),
            Register::new(3.5),
        ]);
        l1.push(neuron);
    }

    let mut layers = Vec::new();
    layers.push(l0);
    layers.push(l1);

    let network = Network {
        nr_inputs: 4,
        nr_outputs: 3,
        time_step_duration_us: 100.0,
        layers,
    };

    let input = vec![
        vec![true, false, false, true, true, false, false, true, true],
        vec![false, false, false, true, false, false, true, true, true],
        vec![false, true, true, true, false, true, true, false, true],
        vec![true, false, false, false, true, false, false, false, true],
    ];

    let output = network.run(input).unwrap();

    for row in &output {
        println!("{:?}", row);
    }
}
