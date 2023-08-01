use crate::neuron::{Message, Neuron};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

pub struct Network {
    pub time_step_duration_us: f64, // time delta between samples (us)
    pub layers: Vec<Vec<Neuron>>,
}

impl Network {
    /// creates a new network
    pub fn new(time_step_duration_us: f64) -> Self {
        Network {
            time_step_duration_us,
            layers: Vec::new(),
        }
    }

    pub fn feed_input(&mut self, input: Vec<Vec<bool>>) -> Result<Vec<Vec<bool>>, ()> {
        /*Number of input nodes (i.e. nr. of rows of the input matrix) */
        let n_input_nodes = input.len();
        /*Number of time slots (i.e. nr. of columns of the input matrix) */
        let n_time_steps = input[0].len();
        /*Number of layers in the network ( = number of threads to be spawn) */
        let n_network_layers = self.layers.len();

        /*Vector to contain handles to wait threads termination */
        let mut thread_handles = Vec::<JoinHandle<()>>::new();

        /*Channels to be used to receive pulses from previous layer and to send
        pulses to the following layer */
        let (input_tx, mut input_rx) = mpsc::channel();
        let (mut output_tx, mut output_rx) = mpsc::channel();

        /*Number of output nodes */
        let n_output_nodes = self.layers.last().unwrap().len();

        /*Output matrix initialization */
        let mut output: Vec<Vec<bool>> = Vec::new();
        for _ in 0..n_output_nodes {
            let mut v = Vec::<bool>::new();
            for _ in 0..n_time_steps {
                v.push(false);
            }
            output.push(v);
        }

        /*Injecting input inside the network (sent to first layer). All inputs
        for a single time slot are sent, first. Input belonging to different
        time slots are separated by the means of a Message::GoAhead, that signals
        that, for the current time slot, no more input spikes must be provided and so
        the listening layer can start its computations*/
        for step in 0..n_time_steps {
            for input_node in 0..n_input_nodes {
                if input[input_node][step] {
                    let _ = input_tx.send(Message::Pulse(input_node)).unwrap();
                }
            }
            input_tx.send(Message::GoAhead).unwrap();
        }

        /*Starting a thread for each layer */
        for layer in 0..n_network_layers {
            /*Each thread is given possession of a clone of its neurons vector*/
            let layer_neurons = self.layers[layer].clone();
            /*Time step duration (ms) */
            let time_step_duration = self.time_step_duration_us / 1000.0;

            /*Spawning each thread */
            let h = thread::spawn(move || {
                /*Getting the input and output channels */
                let rx_i: Receiver<Message> = input_rx;
                let tx_o: Sender<Message> = output_tx;

                /*Neurons vector */
                let mut neurons = layer_neurons;

                /*For each time step */
                for step in 0..n_time_steps {
                    /*Vector to trace the origin of pulses, needed since each synapse
                    has a different weight. */
                    let mut pulse_sources = Vec::new();

                    /*Reading messages from the channel until a Message::GoAhead is received */
                    while let Message::Pulse(source) = rx_i.recv().unwrap() {
                        pulse_sources.push(source);
                    }

                    /*Ended reading all pulses for current time step */

                    /*Updating neurons state */
                    for i in 0..neurons.len() {
                        let mut neuron = &mut neurons[i];

                        /*Sum of all v_mem 'jumps' caused by pulses*/
                        let mut pulse_contribution = 0.0;

                        for source in &pulse_sources {
                            pulse_contribution += neuron.weights[*source];
                        }

                        /*Membrane potential is updated only if the neuron has
                        received at least a pulse */
                        //if pulse_sources.len() > 0 {
                        /*Update potential */
                        neuron.v_mem = neuron.v_rest
                            + (neuron.v_mem - neuron.v_rest)
                                * (((neuron.last_received_pulse_step as f64 - step as f64)
                                    * time_step_duration
                                    / neuron.tau)
                                    .exp())
                            + pulse_contribution;
                        /*Update last_received_pulse_step */
                        neuron.last_received_pulse_step = step;

                        if layer == n_network_layers - 1 && i == 0 {
                            println!("step: {}, v_mem = {}", step, neuron.v_mem);
                        }
                        //}

                        /*Output spikes generation */
                        if neuron.v_mem > neuron.v_th {
                            tx_o.send(Message::Pulse(i as usize)).unwrap();
                            neuron.v_mem = neuron.v_reset;
                        }
                    }

                    /*time step elaboration complete: all pulses sent */
                    tx_o.send(Message::GoAhead).unwrap();
                }
            });

            /*Pushing thread handle to handles vector */
            thread_handles.push(h);

            /*The receiver part of the output channel becomes the input receiver
            part for the next layer */
            input_rx = output_rx;

            if layer == n_network_layers - 1 {
                break;
            }

            /*A new output channel must be created for the next layer (and thread) */
            (output_tx, output_rx) = mpsc::channel();
        }

        thread_handles.into_iter().for_each(|h| h.join().unwrap());

        let mut step = 0;

        while let Ok(message) = input_rx.recv() {
            match message {
                Message::GoAhead => {
                    step += 1;
                }
                Message::Pulse(source) => {
                    output[source][step] = true;
                }
            }
        }

        return Ok(output);
    }
}
