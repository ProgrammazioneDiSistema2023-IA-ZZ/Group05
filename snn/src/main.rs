mod neuron;
mod network;
use crate::neuron::Neuron;
use crate::network::Support;

fn main(){

    let mut n=Neuron::default();
    println!("creato neurone {n:?}");

    let impulse=n.update(vec![0.1,0.01],vec![false,true]);
    println!("c'è stato un impulso in uscita: {impulse}");
    println!("neurone ora è {n:?}");
    let impulse=n.update(vec![0.1,0.01],vec![false,false]);
    println!("c'è stato un impulso in uscita: {impulse}");
    println!("neurone ora è {n:?}");


    let path= "../network_topology.json";
    let letto= Support::from_json(path);
    println!("ho letto {:?}",letto);
    

}