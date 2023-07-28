use std::{
    sync::{Arc, Weak},
    time::Instant,
};
use tokio::sync::mpsc::{Receiver, Sender};

///includes all configurable electrical parameters for a Neuron
#[derive(Clone, Copy)]
pub struct NeuronParameters {
    v_th: f64,    //threshold potential (mV)
    v_rest: f64,  //potential at rest (mV)
    v_reset: f64, //reset potential after generating pulses (mV)
    tau: f64,     //time constant (ms)
}

impl Default for NeuronParameters {
    fn default() -> Self {
        Self {
            v_th: -55.0,    //mV
            v_rest: -70.0,  //mV
            v_reset: -70.0, //mV
            tau: 10.0,      //ms
        }
    }
}

impl NeuronParameters {
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64) -> Self {
        Self {
            v_th,
            v_rest,
            v_reset,
            tau,
        }
    }
}

/// (Empty) struct used to represent a Pulse which can be sent over
/// a channel from one Neuron to another
pub struct Pulse {
    amplitude: f64, //weight of edge on which Pulse is sent
}

impl Pulse {
    ///constructor for Pulse
    pub fn new(amplitude: f64) -> Pulse {
        Pulse { amplitude }
    }
}

///Synapse struct represents an Edge inside the SNN graph. The source node
/// is implicitly the neuron holding the Synapse itself, while the struct
/// contains a pointer to the destination Neuron and a weight.
/// The pointer is of type 'Weak' instead of 'Arc' in order to prevent
/// possible memory leaks due to the presence of loops inside the SNN.
pub struct Synapse {
    destination_neuron: Weak<Neuron>, //target neuron
    weight: f64,                      //synapse weight
}

impl Synapse {
    ///synapse constructor
    pub fn new(destination_neuron: Weak<Neuron>, weight: f64) -> Self {
        Synapse {
            destination_neuron,
            weight,
        }
    }
}

pub struct Neuron {
    id: i32,                                   //unique neuron identifier
    layer: i32,                                //SNN layer to which the neuron belongs
    parameters: NeuronParameters,              //all configurable parameters
    v_mem: f64,                                //membrane potential
    last_received_pulse: Instant,              //time instant when last pulse was received
    channel: (Sender<Pulse>, Receiver<Pulse>), //mpsc channel to receive Pulse structs from other neurons
    synapses_list: Vec<Synapse>,               //list of all adjacencies of the neuron
}

impl Neuron {
    ///construct a new neuron providing id and required electrical parameters
    pub fn new(id: i32, layer: i32, parameters: NeuronParameters) -> Self {
        Neuron {
            id,
            layer,
            v_mem: parameters.v_rest,
            parameters,
            last_received_pulse: Instant::now(),
            channel: tokio::sync::mpsc::channel(100),
            synapses_list: vec![],
        }
    }

    ///method to add a new synapse to a node
    pub fn add_synapse(&mut self, synapse: Synapse) {
        self.synapses_list.push(synapse);
    }

    ///get the Neuron ID
    pub fn get_id(&self) -> i32 {
        self.id
    }

    ///get the Neuron layer
    pub fn get_layer(&self) -> i32 {
        self.layer
    }

    ///get the Neuron v_mem (its output)
    pub fn get_v_mem(&self) -> f64 {
        self.v_mem
    }

    ///get a copy of the Neuron channel sender side, in order to be able
    /// to send Pulse structs to this neuron.
    pub fn get_channel(&self) -> Sender<Pulse> {
        self.channel.0.clone()
    }

    ///method to send pulses to each adjacent Neuron and put v_mem to reset potential
    fn send_pulses(&mut self) {
        for synapse in &(self.synapses_list) {
            let channel = synapse.destination_neuron.upgrade().unwrap().get_channel();
            let pulse = Pulse::new(synapse.weight);
            let _ = channel.blocking_send(pulse);
        }
        self.v_mem = self.parameters.v_reset;
    }

    ///async method to activate the Neuron; it returns an anonymous Future object
    /// which must be run as a task by Tokio runtime (actually executing this method
    /// is responsibility of the SNN data structure)
    pub async fn run(&mut self) {
        // endless loop in which the neuron must:
        // 1. Read from async channel (returns control to other neurons while waiting for input)
        // 2. Update its own status (v_mem and last_received_pulse)
        // 3. If v_mem > v_th send pulses to all its adjacencies (synapses) and reset v_mem
        loop {
            //read from async channel: if nothing to read, then async method
            //execution is suspended, so that other Neurons can execute, instead.
            let message = self.channel.1.recv().await;

            //when a pulse is received
            if let Some(pulse) = message {
                //compute (t_s - t_s-1)
                let now = Instant::now();
                let time_diff = ((now - self.last_received_pulse).as_micros() as f64) * 1e-3;

                //update the time of last received pulse
                self.last_received_pulse = now;

                //update v_mem
                self.v_mem = self.parameters.v_rest
                    + (self.v_mem - self.parameters.v_rest)
                        * (-(time_diff) / self.parameters.tau).exp()
                    + pulse.amplitude;

                //if v_mem > v_th send pulses (and reset v_mem)
                if self.v_mem > self.parameters.v_th {
                    self.send_pulses();
                }
            }
        }
    }
}
