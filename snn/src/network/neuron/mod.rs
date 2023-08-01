/// The Neuron struct represents a neuron of the spiking neural network.
/// A neuron is characterized by a series of parameters which describe its
/// electrical behavior (v_rest, v_reset, v_th, tau).
///
/// Furthermore, each neuron keeps two state variables corresponding to the
/// membrane potential the neuron had during the last time step it received a
/// pulse, and the time step itself.
///
/// Lastly, each neuron keeps a Vec in which the 'i'th cell contains the value of
/// the weight assigned to the synapse going from the 'i'th neuron in the
/// previous layer to that neuron.
#[derive(Clone)]
pub struct Neuron {
    pub v_th: f64, // (mV) threshold voltage: if v_mem goes upper than this value, then a spike is produced as output
    pub v_rest: f64, // (mV) when neuron is not stimulated by pulses, v_mem decreases exponentially towards this value
    pub v_reset: f64, // (mV) when a pulse is produced, v_mem is reset to this value
    pub tau: f64, // (ms) time constant for exponential v_mem decrease in absence of received pulses
    pub v_mem: f64, // (mV) membrane potential after receiving last pulse
    pub last_received_pulse_step: usize, // discrete time step when last pulse was received
    pub weights: Vec<f64>, // weights of each synapse going from the 'i'th neuron of the previous layer to this neuron
}

impl Default for Neuron {
    /// returns a neuron using default electrical parameters
    fn default() -> Self {
        Neuron {
            v_th: -55.0,
            v_rest: -70.0,
            v_reset: -70.0,
            tau: 10.0,
            v_mem: -70.0,
            last_received_pulse_step: 0,
            weights: Vec::new(),
        }
    }
}

impl Neuron {
    /// returns a new neuron having the specified electrical parameters values
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64) -> Self {
        Neuron {
            v_th,
            v_rest,
            v_reset,
            tau,
            v_mem: v_rest,
            last_received_pulse_step: 0,
            weights: Vec::new(),
        }
    }
}

/// A Message can be sent from a layer to another layer in order to transfer
/// pulses or control messages to achieve synchronization between adjacent layers.

pub enum Message {
    // a pulse keeps the index of the Neuron which produced it
    Pulse(usize),
    // notify the following layer that all pulses have been delivered for that time step
    GoAhead,
}
