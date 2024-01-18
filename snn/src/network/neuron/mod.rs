use crate::register::Register;

/// The Neuron struct represents a neuron of the spiking neural network.
/// A neuron is characterized by a series of parameters which describe its
/// electrical behavior (v_rest, v_reset, v_th, tau).
///
/// Furthermore, each neuron keeps two state variables corresponding to the
/// membrane potential the neuron had during the last time step it received a
/// pulse, and the time step itself.
///
/// Each neuron also keeps:
///
/// - a Vec in which the 'i'th cell contains the value of the weight assigned
/// to the synapse going from the 'i'th neuron in the previous layer to that neuron.
/// - a Vec in which the 'i'th cell contains the value of the weight assigned
/// to the internal synapse going from the 'i'th neuron in the SAME layer to that neuron.
/// (synapse going from a node to itself can be assigned weight 0.0)
#[derive(Clone)]
pub struct Neuron {
    pub v_th: Register, // (mV) threshold voltage: if v_mem goes upper than this value, then a spike is produced as output
    pub v_rest: Register, // (mV) when neuron is not stimulated by pulses, v_mem decreases exponentially towards this value
    pub v_reset: Register, // (mV) when a pulse is produced, v_mem is reset to this value
    pub tau: f64, // (ms) time constant for exponential v_mem decrease in absence of received pulses
    pub v_mem: Register, // (mV) membrane potential after receiving last pulse
    pub last_received_pulse_step: usize, // discrete time step when last pulse was received
    pub weights: Vec<Register>, // weights of each synapse going from the 'i'th neuron of the previous layer to this neuron
    pub internal_weights: Vec<Register>, //weights of synapses internal to layer
}

impl Default for Neuron {
    /// returns a neuron using default electrical parameters
    fn default() -> Self {
        Neuron {
            v_th: Register::new(-55.0),
            v_rest: Register::new(-70.0),
            v_reset: Register::new(-70.0),
            tau: 10.0,
            v_mem: Register::new(-70.0),
            last_received_pulse_step: 0,
            weights: Vec::new(),
            internal_weights: Vec::new(),
        }
    }
}

impl Neuron {
    /// returns a new neuron having the specified electrical parameters values
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64) -> Self {
        Neuron {
            v_th: Register::new(v_th),
            v_rest: Register::new(v_rest),
            v_reset: Register::new(v_reset),
            tau,
            v_mem: Register::new(v_rest),
            last_received_pulse_step: 0,
            weights: Vec::new(),
            internal_weights: Vec::new(),
        }
    }

    /// Set weights for synapses external to the current layer. Cell 'i' in the weights Vec
    /// represents the weight assigned to the synapse going from the Neuron with index
    /// 'i' in the previous layer to the current Neuron.
    /// 'Weights' Vec must have as many elements as the number of Neurons in the previous
    /// layer.
    pub fn set_weights(&mut self, weights: Vec<f64>) {
        self.weights = weights
            .into_iter()
            .map(|w| Register::new(w))
            .collect::<Vec<Register>>();
    }

    /// Set weights for synapses internal to the current layer. Cell 'i' in the internal_weights
    /// Vec represents the weight associated to the INTERNAL synapse going from the Neuron with
    /// index 'i' of the SAME layer as the current Neuron, to the current Neuron itself.
    /// internal_weights Vec must have as many elements as the number of Neurons in the current
    /// layer. The weight of the internal synapse going from a node to itself should be set to 0.0.
    pub fn set_internal_weights(&mut self, internal_weights: Vec<f64>) {
        self.internal_weights = internal_weights
            .into_iter()
            .map(|w| Register::new(w))
            .collect::<Vec<Register>>();
    }

    /// simulate the reception of a series of Pulses on the input synapses for the Neuron
    /// which causes a change in the Membrane Potential. If this potential goes beyond the
    /// threshold (v_th), then the function returns true, simulating the emission of a
    /// Pulse, otherwise it returns false.
    pub fn feed_pulses(
        &mut self,
        pulse_sources: &Vec<usize>,
        time_step: usize,
        time_step_duration_ms: f64,
    ) -> bool {
        // reading required parameters and states of the Neuron from Registers. If any
        // Register is affected by a Damage, this could alter the read value.
        let v_th = self.v_th.read_value(Some(time_step)).unwrap();
        let v_rest = self.v_rest.read_value(Some(time_step)).unwrap();
        let v_reset = self.v_reset.read_value(Some(time_step)).unwrap();
        let old_v_mem = self.v_mem.read_value(Some(time_step)).unwrap();

        // computing v_mem increment contribution due to pulses
        let pulses_contribution = self.get_pulses_contribution(pulse_sources, time_step);

        // computing new Membrane Potential
        let v_mem = v_rest
            + (old_v_mem - v_rest)
                * ((self.last_received_pulse_step as f64 - time_step as f64)
                    * time_step_duration_ms
                    / self.tau)
                    .exp()
            + pulses_contribution;

        // updating last_received_pulse_step
        self.last_received_pulse_step = time_step;

        //comparing v_mem to threshold
        if v_mem >= v_th {
            // The Neuron fires: Membrane potential must be reset
            self.v_mem.write_value(v_reset);
            return true;
        }

        // The Neuron does not fire: write new v_mem to the register
        self.v_mem.write_value(v_mem);
        return false;
    }

    /// simulate loss of membrane potential for a Neuron when other Neurons of the same
    /// layer 'fire' at the preceding time step. The amount of potential loss depends on
    /// the content of the internal_weights Vec
    pub fn inhibite_after_pulses_emission(
        &mut self,
        pulse_sources: &Vec<usize>,
        time_step: usize,
        time_step_duration_ms: f64,
    ) {
        // reading required parameters and states of the Neuron from Registers. If any
        // Register is affected by a Damage, this could alter the read value.
        let v_rest = self.v_rest.read_value(Some(time_step)).unwrap();
        let old_v_mem = self.v_mem.read_value(Some(time_step)).unwrap();

        // computing v_mem inhibitive contribution
        let inhibitive_contribution = self.get_inhibitive_contribution(pulse_sources, time_step);

        // computing new Membrane Potential
        let v_mem = v_rest
            + (old_v_mem - v_rest)
                * ((self.last_received_pulse_step as f64 - time_step as f64)
                    * time_step_duration_ms
                    / self.tau)
                    .exp()
            + inhibitive_contribution;

        // updating last_received_pulse_step
        self.last_received_pulse_step = time_step;

        // writing new v_mem into Register
        self.v_mem.write_value(v_mem);
    }

    ///compute pulse contribution to v_mem, based on the stored weights
    fn get_pulses_contribution(&self, pulse_sources: &Vec<usize>, time_step: usize) -> f64 {
        let mut contribution = 0.0;
        for source_index in pulse_sources {
            // read weight from Register
            let weight = self.weights[*source_index]
                .read_value(Some(time_step))
                .unwrap();
            // add contribution
            contribution += weight;
        }

        return contribution;
    }

    ///compute inhibitive contribution to v_mem, based on the stored internal weights
    fn get_inhibitive_contribution(&self, pulse_sources: &Vec<usize>, time_step: usize) -> f64 {
        let mut contribution = 0.0;
        for source_index in pulse_sources {
            // read internal weight from Register
            let internal_weight = self.internal_weights[*source_index]
                .read_value(Some(time_step))
                .unwrap();
            // add contribution
            contribution += internal_weight;
        }

        return contribution;
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
