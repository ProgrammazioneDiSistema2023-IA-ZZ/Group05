use snn::network;

fn main() {
    let mut network = network::json::load_from_file(
        "src\\snn_data.json",
    );
    let input = vec![
        vec![true, false, true, false, false],
        vec![true, true, true, false, true],
        vec![false, false, false, true, false],
        vec![true, false, false, true, true],
        vec![false, true, true, false, false],
        vec![false, false, false, false, true],
    ];
    let output = network.run(input).unwrap();

    output
        .iter()
        .enumerate()
        .for_each(|(ind, v)| println!("out{}: {:?}", ind, v));
}
