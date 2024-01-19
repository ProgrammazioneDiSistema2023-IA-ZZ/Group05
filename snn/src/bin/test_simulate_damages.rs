use snn::network::{self, DamageModel, FaultyElement};
use std::fs::write;

fn main() {
    let network = network::json::load_from_file("sources\\snn_data.json");
    let input = vec![
        vec![true, true, false, true, false, true, true, true, true, true],
        vec![true, true, true, true, true, true, true, true, true, true],
        vec![true, true, false, true, true, true, true, true, true, true],
        vec![true, true, true, true, true, true, true, true, true, false],
        vec![true, true, true, true, true, true, true, true, true, true],
        vec![true, true, true, true, true, true, true, true, true, true],
    ];

    let faulty_elements = vec![
        FaultyElement::Weights,
        FaultyElement::Thresholds,
        FaultyElement::MembranePotentials,
        FaultyElement::ResetPotentials,
        FaultyElement::PotentialsAtRest,
        FaultyElement::Comparator,
        FaultyElement::Adder,
        FaultyElement::Multiplier,
        FaultyElement::Divider,
    ];
    let damage_type = DamageModel::TransientBitFlip;
    let output = network
        .simulate(faulty_elements, damage_type, 10000, input)
        .unwrap();

    /* output_matrix
    .iter()
    .enumerate()
    .for_each(|(ind, v)| println!("out{}: {:?}", ind, v)); */

    let output_matrix = &output.diffs;

    for i in 0..3 {
        print!("out{i}: ");
        for j in 0..10 {
            print!("{:?} ", output_matrix[i][j].diff_count);
        }
        println!();
    }

    // Write full simulation result to file
    let output_path = "output\\simulation_output.json";
    let serialized = serde_json::to_string(&output).unwrap();
    let _ = write(output_path, serialized).unwrap();
}
