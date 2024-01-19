use crate::register::Register;

use super::NeuronModel;

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
    pub tau: Register, // (ms) time constant for exponential v_mem decrease in absence of received pulses
    pub v_mem: Register, // (mV) membrane potential after receiving last pulse
    pub last_received_pulse_step: usize, // discrete time step when last pulse was received
    pub weights: Vec<Register>, // weights of each synapse going from the 'i'th neuron of the previous layer to this neuron
    pub internal_weights: Vec<Register>, //weights of synapses internal to layer
    pub add_reg: Register,      // register which contains the output of adder
    pub mul_reg: Register,      // register which contains the output of multiplier
    pub cmp_reg: Register,      // register which contains the output of comparator
    pub div_reg: Register,      // register which contains the output of divider
}

impl Default for Neuron {
    /// returns a neuron using default electrical parameters
    fn default() -> Self {
        Neuron {
            v_th: Register::new(-55.0),
            v_rest: Register::new(-70.0),
            v_reset: Register::new(-70.0),
            tau: Register::new(10.0),
            v_mem: Register::new(-70.0),
            last_received_pulse_step: 0,
            weights: Vec::new(),
            internal_weights: Vec::new(),
            add_reg: Register::new(0.0),
            mul_reg: Register::new(0.0),
            cmp_reg: Register::new(0.0),
            div_reg: Register::new(0.0),
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
            tau: Register::new(tau),
            v_mem: Register::new(v_rest),
            last_received_pulse_step: 0,
            weights: Vec::new(),
            internal_weights: Vec::new(),
            add_reg: Register::new(0.0),
            mul_reg: Register::new(0.0),
            cmp_reg: Register::new(0.0),
            div_reg: Register::new(0.0),
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
        neuron_model: NeuronModel,
    ) -> bool {
        self.update_membrane_potential(
            pulse_sources,
            time_step,
            time_step_duration_ms,
            neuron_model,
            PulseContributionMode::Excitatory,
        );

        // updating last_received_pulse_step
        self.last_received_pulse_step = time_step;

        //comparing v_mem to threshold
        Register::cmp(self.v_mem, self.v_th, &mut self.cmp_reg, time_step);
        if self.cmp_reg.read_value(Some(time_step)).unwrap() >= 0.0 {
            // The Neuron fires: Membrane potential must be reset
            self.v_reset.copy_to(&mut self.v_mem, time_step);
            return true;
        }

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
        neuron_model: NeuronModel,
    ) {
        self.update_membrane_potential(
            pulse_sources,
            time_step,
            time_step_duration_ms,
            neuron_model,
            PulseContributionMode::Inhibitive,
        );

        // updating last_received_pulse_step
        self.last_received_pulse_step = time_step;
    }

    ///compute pulse contribution to v_mem, based on the stored weights
    fn get_pulses_contribution(&self, pulse_sources: &Vec<usize>, time_step: usize) -> Register {
        let mut add_reg = self.add_reg;
        add_reg.write_value(0.0);
        for source_index in pulse_sources {
            Register::add(
                add_reg,
                self.weights[*source_index],
                &mut add_reg,
                time_step,
            );
        }

        return add_reg;
    }

    ///compute inhibitive contribution to v_mem, based on the stored internal weights
    fn get_inhibitive_contribution(
        &self,
        pulse_sources: &Vec<usize>,
        time_step: usize,
    ) -> Register {
        let mut add_reg = self.add_reg;
        add_reg.write_value(0.0);
        for source_index in pulse_sources {
            Register::add(
                add_reg,
                self.internal_weights[*source_index],
                &mut add_reg,
                time_step,
            );
        }

        return add_reg;
    }

    /// Update membrane potential according to the provided neuron model
    fn update_membrane_potential(
        &mut self,
        pulse_sources: &Vec<usize>,
        time_step: usize,
        time_step_duration_ms: f64,
        neuron_model: NeuronModel,
        pulse_contribution_mode: PulseContributionMode,
    ) {
        // computing v_mem contribution due to pulses
        let pulses_contribution = match pulse_contribution_mode {
            PulseContributionMode::Excitatory => {
                self.get_pulses_contribution(pulse_sources, time_step)
            }
            PulseContributionMode::Inhibitive => {
                self.get_inhibitive_contribution(pulse_sources, time_step)
            }
        };

        // computing new Membrane Potential

        let mut pulses_contrib_reg = Register::new(0.0);
        Register::add(
            self.v_mem,
            pulses_contribution,
            &mut self.add_reg,
            time_step,
        );
        self.add_reg.copy_to(&mut pulses_contrib_reg, time_step);

        match neuron_model {
            NeuronModel::LeakyIntegrateAndFire => {
                // computing v_mem - v_rest
                let mut vm_vr = Register::new(0.0);
                Register::sub(self.v_mem, self.v_rest, &mut self.add_reg, time_step);
                self.add_reg.copy_to(&mut vm_vr, time_step);

                // computing last_received_pulse_step - time_step
                let diff_steps =
                    Register::new(self.last_received_pulse_step as f64 - time_step as f64);

                // computing exp argument
                let mut exp_arg = Register::new(0.0);
                Register::mult(
                    diff_steps,
                    Register::new(time_step_duration_ms),
                    &mut self.mul_reg,
                    time_step,
                );
                Register::div(self.mul_reg, self.tau, &mut self.div_reg, time_step);
                self.div_reg.copy_to(&mut exp_arg, time_step);

                // performing exp
                let exp_res = Register::new(exp_arg.read_value(Some(time_step)).unwrap().exp());

                // computing exp * (v_mem - v_rest)
                let mut decay_part = Register::new(0.0);
                Register::mult(exp_res, vm_vr, &mut self.mul_reg, time_step);
                self.mul_reg.copy_to(&mut decay_part, time_step);

                // computing decay_part + pulses_contrib_reg
                Register::add(decay_part, pulses_contrib_reg, &mut self.add_reg, time_step);
                self.add_reg.copy_to(&mut self.v_mem, time_step);
            }
            NeuronModel::IntegrateAndFire => {
                pulses_contrib_reg.copy_to(&mut self.v_mem, time_step);
            }
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

/// Specify if neuron is receiving pulses on excitatory or inhibitive layer
enum PulseContributionMode {
    Excitatory,
    Inhibitive,
}
