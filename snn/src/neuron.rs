use std::sync::Arc;
#[derive(Debug)]
pub struct Neuron{
    v_mem :f64,
    v_th :f64, //configurabile
    v_rest :f64, //configurabile
    v_reset :f64, //configurabile
    tau :f64, //configurabile
    //output :bool
}

pub struct Synapsis{
    weight: f64,
    source_neuron: Arc<Neuron>,
    dest_neuron: Arc<Neuron>,
}

impl Default for Neuron {
    fn default() -> Self{
        Neuron{
            v_mem: -7e-2,
            v_th: -5.5e-2,
            v_rest: -7e-2,
            v_reset: -7e-2,
            tau: 1e-2,
            //output:false
        }
    }
}
impl Neuron{
    
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64) -> Self{
        Neuron{
            v_mem: v_rest,
            v_th: v_th,
            v_rest: v_rest,
            v_reset: v_reset,
            tau: tau,
            //output:false
        }
    }

    //to call periodically with a sampling time that grows for every impulse not received
    pub fn update(&mut self, weights: Vec<f64>, is_active: Vec<bool>, ts: f64)->bool{

        //using LIF: v_mem(ts)= v_rest+[v_mem(ts-1)-v_rest]*e^((ts-ts-1)/tau)  +SUM(0->N){si*wi}
        self.v_mem=self.v_rest+(self.v_mem-self.v_rest)*(-ts/self.tau).exp();
        println!("valore potenziale senza pesi: {}",self.v_mem);
        if weights.len()==is_active.len(){
            for i in 0..weights.len(){
                self.v_mem+=( if is_active[i]==true {1.0} else {0.0} )*weights[i];
            }
            println!("valore potenziale con pesi: {}",self.v_mem);    
        }else{
            panic!("wrong dimensions for weights and is_active")
        }


        if self.v_mem >= self.v_th{ //spiking
            self.v_mem=self.v_reset;
            return true; //output impulse
        }
        false //threshold not reached

    }
}

impl Synapsis{
    pub fn new(weight: f64, source_neuron: Arc<Neuron>, dest_neuron: Arc<Neuron>) -> Self{
        Synapsis{
            weight: weight,
            source_neuron: source_neuron,
            dest_neuron: dest_neuron,
        }
    }
}