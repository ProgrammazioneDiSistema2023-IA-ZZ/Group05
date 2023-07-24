
pub struct Neuron{
    v_mem :f64, //configurabile
    v_th :f64, //configurabile
    v_rest :f64, //configurabile
    output :bool
}

pub struct Synapsis{
    weight:f64,
    propagation_time:f64
}

impl Neuron{
    pub fn default() -> Self{
        Neuron{
            v_mem:-7e-2,
            v_th:-5.5e-2,
            v_rest:-7e2,
            output:false
        }
    }
    pub fn new(v_mem:f64,v_th:f64,v_rest:f64) -> Self{
        Neuron{
            v_mem,
            v_th,
            v_rest,
            output:false
        }
    }
}

impl Synapsis{
    pub fn new() -> Self{
        Synapsis{
            weight:1.0,
            propagation_time:0.3e-3
        }
    }
}