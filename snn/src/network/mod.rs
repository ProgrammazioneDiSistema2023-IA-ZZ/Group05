use crate::network::neuron::{Message, Neuron};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

pub mod json;
pub mod neuron;

/** The struct Network represents a Spiking Neural Network.
All the Neurons inside the network belong to a layer and
each layer collects its own Neurons inside a Vec. The set
of all Vec representing layers is then collected inside an
outer Vec.

The Network evolves temporally in discrete time steps, whose
length can be set at network creation (expressed in microseconds). */
pub struct Network {
    pub nr_inputs: usize,
    pub nr_outputs: usize,
    pub time_step_duration_us: f64, /* Time step duration*/
    pub layers: Vec<Vec<Neuron>>,   /* Vec collecting layers (other Vecs) */
}

impl Network {
    /** Create a network */
    pub fn new(time_step_duration_us: f64, nr_inputs: usize, nr_outputs: usize) -> Self {
        Network {
            nr_inputs,
            nr_outputs,
            time_step_duration_us,
            layers: Vec::new(),
        }
    }

    /** Add a layer to the network */
    pub fn add_layer(&mut self, layer: Vec<Neuron>) {
        self.layers.push(layer);
    }

    /** Get output nodes number */
    pub fn get_outputs_number(&self) -> Result<usize, ()> {
        if self.layers.len() == 0 {
            return Err(());
        }
        return Ok(self.layers.last().unwrap().len());
    }
    /** Matrix of output_neurons x time_steps */
    pub fn create_output_matrix(&self, time_steps: usize) -> Vec<Vec<bool>> {
        let mut output = Vec::<Vec<bool>>::new();
        for _ in 0..self.get_outputs_number().unwrap() {
            output.push(vec![false; time_steps]);
        }
        output
    }

    /** Referring to a channel between two layers, listen up to any message.
     * If an impulse arrives, make note; go on with the next time step only if a message GoAhead arrives.*/
    pub fn write_results(rx: Receiver<Message>, output: &mut Vec<Vec<bool>>) {
        let mut step = 0;

        while let Ok(message) = rx.recv() {
            match message {
                Message::GoAhead => {
                    step += 1;
                }
                Message::Pulse(source) => {
                    output[source][step] = true;
                }
            }
        }
    }

    /** Run the network using the provided input(Matrix with input_neurons x time_steps) */
    pub fn run(&mut self, input: Vec<Vec<bool>>) -> Result<Vec<Vec<bool>>, ()> {
       
        /*Number of input nodes ( = nr. rows of the input matrix) */
        let n_input_nodes = input.len();
        /*Number of time slots ( =  nr. of columns of the input matrix) */
        let n_time_steps = input[0].len();

        /*Number of layers in the network ( = number of threads to be spawn) */
        let n_network_layers = self.layers.len();

        /*Vector to contain handles to wait threads termination */
        let mut thread_handles = Vec::<JoinHandle<()>>::new();

        /*Channels to be used to receive pulses from previous layer and to send
        pulses to the following layer. Input and Output names refer to the relative
        'position' of the channel to the current layer: [input] --> {layer} --> [output]  */
        let (input_tx, mut input_rx) = mpsc::channel();
        let (mut output_tx, mut output_rx) = mpsc::channel();

        /*Output matrix initialization */
        let mut output = self.create_output_matrix(n_time_steps);

        /*Injecting input inside the network (to first layer). All inputs
        for a single time slot are sent, first. Input belonging to different
        time slots are separated by the means of a Message::GoAhead, that signals
        that, for the current time slot, no more input spikes must be provided and so
        the listening layer can start its computation*/
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
            /*Each thread is given possession of a clone of its own neurons vector*/
            let layer_neurons = self.layers[layer].clone();
            /*Time step duration (ms) */
            let time_step_duration = self.time_step_duration_us / 1000.0;

            /*Spawning each thread */
            let h = thread::Builder::new()
                .name(format!("layer {}", layer))
                .spawn(move || {
                    /*Getting the input and output channels */
                    let rx_i: Receiver<Message> = input_rx;
                    let tx_o: Sender<Message> = output_tx;

                    /*Neurons vector name change*/
                    let mut neurons = layer_neurons;
                    /*Vectors used for modeling internal pulses */
                    let mut internal_impulses:Vec<usize>= Vec::new();
                    let mut internal_impulses_next:Vec<usize>=Vec::new();

                    /*For each time step */
                    for step in 0..n_time_steps {

                        /*Vector to trace the origin of pulses, needed since each synapse
                        has a different weight. */
                        let mut pulse_sources = Vec::new();
                        while let Ok(message) = rx_i.recv() {
                            if let Message::Pulse(source) = message {
                                pulse_sources.push(source);
                            } else {
                                break;
                            }
                        }

                        /*Ended reading all pulses for current time step */

                        /*Updating neurons state */
                        let neurons_of_layer = neurons.len();
                        for i in 0..neurons_of_layer {
                            let mut neuron = &mut neurons[i];

                            /*Sum of all v_mem 'jumps' caused by weighted pulses*/
                            let mut pulse_contribution = 0.0;

                            //println!("layer: {}, neuron: {}, step: {}", layer, i, step);
                            for source in &pulse_sources {
                                pulse_contribution += neuron.weights[*source];
                            }
                            for source in &internal_impulses{
                                /*take note of internal impulses of the other neuorons of the layer */
                                if *source!=i{ pulse_contribution+= neuron.internal_weights[*source]};
                            }

                            /*Membrane potential is updated only if the neuron has
                            received at least a pulse */
                            if pulse_contribution!=0.0 {   //CONSIDERIAMO ANCHE QUELLO INTERNO? pulse_sources.len() > 0
                            /*Update potential using LIF*/
                            neuron.v_mem = neuron.v_rest
                                + (neuron.v_mem - neuron.v_rest)
                                    * (((neuron.last_received_pulse_step as f64 - step as f64)
                                        * time_step_duration
                                        / neuron.tau)
                                        .exp())
                                + pulse_contribution;
                            /*Update last_received_pulse_step */
                            neuron.last_received_pulse_step = step;

                            /* If v_mem exceeds threshold */
                            if neuron.v_mem > neuron.v_th {
                                /* Generate spikes as output */
                                tx_o.send(Message::Pulse(i as usize)).unwrap();
                                internal_impulses_next.push(i as usize);
                                /* Reset v_mem */
                                neuron.v_mem = neuron.v_reset;
                                /* Decrease v_mem for neurons of same layer*/
                                /*let mut ind = 0;
                                for j in 0..neurons_of_layer {
                                    if j != i {
                                        neurons[j].v_mem += neurons[i].internal_weights[ind];
                                        ind += 1;
                                    }
                                }*/
                            }
                            }
                        }

                    /*time step elaboration complete: all pulses sent */
                    tx_o.send(Message::GoAhead).unwrap();
                    internal_impulses=internal_impulses_next.clone();
                    internal_impulses_next.clear();
                    }
                });

            /*Pushing thread handle to handles vector */
            thread_handles.push(h.unwrap());

            /*The receiver part of the output channel becomes the input receiver
            part for the next layer. If last layer has just been started, then
            input_rx can be used to read results */
            input_rx = output_rx;

            /*No need to create next output channel if last layer has been started*/
            if layer == n_network_layers - 1 {
                break;
            }

            /*A new output channel must be created for the next layer (and thread) */
            (output_tx, output_rx) = mpsc::channel();
        }//end of layer 

        /*Waiting for all threads to finish */
        thread_handles.into_iter().for_each(|h| {
            h.join().unwrap();
        });

        /*Writing results to output matrix*/
        Self::write_results(input_rx, &mut output);

        return Ok(output);
    }
}
