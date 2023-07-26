use std::sync::Arc;
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