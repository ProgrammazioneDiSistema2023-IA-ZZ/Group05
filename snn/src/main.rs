mod neuron;
use crate::neuron::Neuron;

fn main(){

    let mut n=Neuron::default();
    println!("creato neurone {n:?}");

    let impulse=n.update(vec![0.1,0.01],vec![false,true],0.1);
    println!("c'è stato un impulso in uscita: {impulse}");
    println!("neurone ora è {n:?}");
    let impulse=n.update(vec![0.1,0.01],vec![false,false],0.3);
    println!("c'è stato un impulso in uscita: {impulse}");
    println!("neurone ora è {n:?}");

}