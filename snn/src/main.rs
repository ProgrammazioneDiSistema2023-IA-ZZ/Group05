use clap::Parser;
use snn::network::{self, json};
use snn::network::{DamageModel, FaultyElement};
use std::fs::{self, File};
use std::io::Write;

/// Program to simulate the behaviour of a Spiking Neural Network when
/// some of its components present some damages.
#[derive(Parser)]
struct Args {
    /// json file which describes the network structure.
    #[arg(short, long, default_value_t = String::from("sources\\snn_data.json"))]
    network_json: String,
    /// json output file that will contain the output of the simulation
    #[arg(short, long, default_value_t = String::from("output\\simulation_output.json"))]
    output_file: String,
    /// json file containing the input sequence to be fed to the network
    #[arg(short, long, default_value_t = String::from("sources\\simulation_input.json"))]
    input_file: String,
    /// comma separated list of elements to be damaged
    #[arg(
        short,
        long,
        default_value_t = String::from(
            "weights, thresholds, membrane_potentials, reset_potentials, potentials_at_rest"
        )
    )]
    damaged_elements_list: String,
    /// number of simulations to be performed
    #[arg(short, long, default_value_t = 1000)]
    simulation_iterations: usize,
    /// Damage model to be applied, among stuck_at_0, stuck_at_1,
    /// and transient_bit_flip
    #[arg(short, long, default_value_t = String::from("stuck_at_0"))]
    type_of_damage: String,
}
fn main() {
    // parse arguments
    let args = Args::parse();

    // check existence of network_json file
    check_if_file_exists(&args.network_json);
    // check existence of input_file
    check_if_file_exists(&args.input_file);

    // check damaged_elements_list
    let mut faulty_elements = vec![];
    for element in args.damaged_elements_list.replace(" ", "").split(",") {
        // adding element to FaultyElement Vec
        match element {
            "weights" => {
                faulty_elements.push(FaultyElement::Weights);
            }
            "thresholds" => {
                faulty_elements.push(FaultyElement::Weights);
            }
            "membrane_potentials" => {
                faulty_elements.push(FaultyElement::MembranePotentials);
            }
            "reset_potentials" => {
                faulty_elements.push(FaultyElement::ResetPotentials);
            }
            "potentials_at_rest" => {
                faulty_elements.push(FaultyElement::PotentialsAtRest);
            }
            _ => {
                panic!("{element} is not a valid element!");
            }
        }
    }

    // check damage_model
    let damage_model;
    match args.type_of_damage.as_str() {
        "stuck_at_0" => {
            damage_model = DamageModel::StuckAt0;
        }
        "stuck_at_1" => {
            damage_model = DamageModel::StuckAt1;
        }
        "transient_bit_flip" => {
            damage_model = DamageModel::TransientBitFlip;
        }
        _ => {
            panic!("{} is not a valid damage model!", args.type_of_damage);
        }
    }

    // loading network from file
    let network = network::json::load_from_file(&args.network_json);
    // loading input from file
    let input = json::InputMatrix::load_from_file(&args.input_file).0;

    // start simulation
    let output_matrix = network
        .simulate(
            faulty_elements,
            damage_model,
            args.simulation_iterations,
            input,
        )
        .unwrap();

    let serialized_output_matrix = serde_json::to_string(&output_matrix).expect("Cannot serialize");

    // write results to file
    let mut file = File::create(args.output_file).expect("Cannot open file");
    file.write_all(serialized_output_matrix.as_bytes())
        .expect("Cannot write file");

    // print summed up output to screen
    output_matrix.print();
}

fn check_if_file_exists(file_path: &str) {
    if let Ok(metadata) = fs::metadata(file_path) {
        if !metadata.is_file() {
            panic!("{file_path} is not a valid file!");
        }
    } else {
        panic!("{file_path} does not exist!");
    }
}
