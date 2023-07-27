mod neuron;
use crate::neuron::Neuron;

fn main(){

    let mut n=Neuron::default();
    println!("creato neurone {n:?}");

    let impulse=n.update(vec![0.1,0.01],vec![false,true]);
    println!("c'è stato un impulso in uscita: {impulse}");
    println!("neurone ora è {n:?}");
    let impulse=n.update(vec![0.1,0.01],vec![false,false]);
    println!("c'è stato un impulso in uscita: {impulse}");
    println!("neurone ora è {n:?}");

}