pub mod neuron {
    use std::time::Instant;
    use tokio::sync::mpsc::{Receiver, Sender};

    ///includes all configurable parameters for a neuron

    #[derive(Clone, Copy)]
    pub struct NeuronParameters {
        v_th: f64,    //threshold potential (mv)
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

    pub struct Pulse;

    pub struct Neuron {
        id: i32,                                   //unique neuron identifier
        parameters: NeuronParameters,              //all configurable parameters
        v_mem: f64,                                //membrane potential
        last_received_pulse: Instant,              //time instant when last pulse was received
        channel: (Sender<Pulse>, Receiver<Pulse>), //mpsc channel to receive Pulse structs from other neurons
    }

    impl Default for Neuron {
        fn default() -> Self {
            Neuron {
                id: -1, //for testing purposes only (invalid ID)
                parameters: NeuronParameters::default(),
                v_mem: -70.0, //mV
                last_received_pulse: Instant::now(),
                channel: tokio::sync::mpsc::channel(100),
            }
        }
    }

    impl Neuron {
        ///construct a new neuron providing id and required parameters
        pub fn new(id: i32, parameters: NeuronParameters) -> Self {
            Neuron {
                id: id,
                v_mem: parameters.v_rest,
                parameters: parameters,
                last_received_pulse: Instant::now(),
                channel: tokio::sync::mpsc::channel(100),
            }
        }

        ///get a copy of the Neuron channel sender side, in order to be able
        /// to send Pulse structs to this neuron.
        pub fn get_channel(&self) -> Sender<Pulse> {
            self.channel.0.clone()
        }

        ///activate neuron lifecycle
        pub async fn run(&mut self) {
            loop {
                let val = self.channel.1.recv().await;
                match val {
                    Some(pulse) => {
                        println!("Pulse received!!!");
                    }
                    None => {
                        println!("Warning: no pulse...");
                    }
                };
            }
        }
    }
}
