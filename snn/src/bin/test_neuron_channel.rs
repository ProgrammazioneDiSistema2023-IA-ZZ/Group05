use std::time::Duration;

use snn::neuron::{self, Pulse};
use tokio::join;

#[tokio::main]
async fn main() {
    /*test neuron communication through channel */

    let parameters = neuron::NeuronParameters::default();

    let mut node = neuron::Neuron::new(0, 0, parameters);
    let channel = node.get_channel();
    let h1 = tokio::spawn(async move {
        node.run().await;
    });

    let h2 = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(3)).await;
            let _ = channel.send(Pulse::new(1.0)).await.unwrap();
        }
    });

    let _ = join!(h1, h2);
}
