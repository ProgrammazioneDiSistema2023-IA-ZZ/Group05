use rand::Rng;
use rand::{seq::SliceRandom, thread_rng};

use crate::network::neuron::{Message, Neuron};
use crate::register::Damage;
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};

pub mod json;
pub mod neuron;

/// Struct to describe damage in detail
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct DamageDetail {
    at_iteration: usize,
    damage_type: FaultyElement,
    at_layer: usize,
    at_neuron: usize,
    at_bit: usize,
}

/// Struct to hold the simulation result
#[derive(Serialize, Deserialize)]
pub struct SimulationResultCell {
    // index of the output neuron which produced the value
    pub output_index: usize,
    // time step when this output was produced
    pub time_step: usize,
    // value resulting from simulation, opposite to the expected value
    pub actual_value: bool,
    // keep track of how many times the output for a certain exit and time step
    // differs from the original simulation
    pub diff_count: usize,
    // keep track of the iterations when the output differs from the original simulation
    pub damage_details: Vec<DamageDetail>,
}

impl SimulationResultCell {
    fn new(row_id: usize, col_id: usize) -> Self {
        SimulationResultCell {
            output_index: row_id,
            time_step: col_id,
            actual_value: false,
            diff_count: 0,
            damage_details: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct SimulationResult {
    pub number_of_iterations: usize,
    pub type_of_damage: DamageModel,
    pub output_without_damages: Vec<Vec<bool>>,
    pub diffs: Vec<Vec<SimulationResultCell>>,
}

impl SimulationResult {
    pub fn print(&self) {
        // print output_without_damages
        println!("Output without damages");
        for (i, row) in self.output_without_damages.iter().enumerate() {
            print!("Out{i}: ");
            for col in row.iter() {
                print!("{} ", col);
            }
            println!("");
        }

        println!("\n");
        println!("Number of recored differences");

        // print diffs
        for (i, row) in self.diffs.iter().enumerate() {
            print!("Out{i}: ");
            for col in row.iter() {
                print!("{} ", col.diff_count);
            }
            println!("");
        }
    }
}

/// enum FaultyElement lists the types of elements in the network which could be
/// potentially subject to damages
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FaultyElement {
    Weights,
    Thresholds,
    MembranePotentials,
    ResetPotentials,
    PotentialsAtRest,
    Comparator,
    Adder,
    Multiplier,
    Divider,
}

/// enum DamageModel is used to specify what kind of damage to the network elements
/// should be simulated
#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DamageModel {
    StuckAt0,
    StuckAt1,
    TransientBitFlip,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum NeuronModel {
    LeakyIntegrateAndFire,
    IntegrateAndFire,
}

/// The struct Network represents a Spiking Neural Network.
/// All the Neurons inside the network belong to a layer and
/// each layer collects its own Neurons inside a Vec. The set
/// of all Vec representing layers is then collected inside an
/// outer Vec.
///
/// The Network evolves temporally in discrete time steps, whose
/// length can be set at network creation (expressed in microseconds).
#[derive(Clone)]
pub struct Network {
    pub nr_inputs: usize,
    pub nr_outputs: usize,
    pub time_step_duration_us: f64, // Time step duration
    pub layers: Vec<Vec<Neuron>>,   // Vec collecting layers (other Vecs)
    pub model: NeuronModel,         // Model used by neurons (e.g. LIF, IF)
}

impl Network {
    /// Create a network
    pub fn new(
        time_step_duration_us: f64,
        nr_inputs: usize,
        nr_outputs: usize,
        model: NeuronModel,
    ) -> Self {
        Network {
            nr_inputs,
            nr_outputs,
            time_step_duration_us,
            layers: Vec::new(),
            model,
        }
    }

    /// Add a layer to the network
    pub fn add_layer(&mut self, layer: Vec<Neuron>) {
        self.layers.push(layer);
    }

    /// Get output nodes number
    pub fn get_outputs_number(&self) -> Result<usize, ()> {
        if self.layers.len() == 0 {
            return Err(());
        }
        return Ok(self.layers.last().unwrap().len());
    }
    /// Matrix of output_neurons x time_steps
    pub fn create_output_matrix(&self, time_steps: usize) -> Vec<Vec<bool>> {
        let mut output = Vec::<Vec<bool>>::new();
        for _ in 0..self.get_outputs_number().unwrap() {
            output.push(vec![false; time_steps]);
        }
        output
    }

    /// Read all pulses produced by the last layer of the SNN from the receiver side of
    /// its output channel and write the result of the SNN computation inside the 'output'
    /// boolean matrix, having dimension (nr. output neurons) x (nr. time steps).
    /// Each row corresponds to a particular output neuron, by index, while each column
    /// correponds to a certain time step. If output[i][j] == true, it means that, at time
    /// step 'j', the output neuron 'i' produced a pulse (false means no pulse, instead).
    pub fn write_results(rx: Receiver<Message>, output: &mut Vec<Vec<bool>>) {
        let mut time_step = 0;

        // read all Messages from the receiver side of the output channel
        while let Ok(message) = rx.recv() {
            match message {
                // if the message is GoAhead, that means that all the pulses for the current
                // time step have been read, and so time step can be incremented (i.e. it is
                // possible to start filling the following column in the output matrix)
                Message::GoAhead => {
                    time_step += 1;
                }
                // if the message is a Pulse, then it contains the index of the neuron
                // of the previous layer which produced the pulse itself, which also matches
                // the index of the row in the output matrix to fill
                Message::Pulse(source_index) => {
                    output[source_index][time_step] = true;
                }
            }
        }
    }

    /// checks if the input provided to the SNN is valid, i.e. if all the rows in the matrix
    /// have the same length (the matrix is actually a Vec of Vecs of booleans)
    fn input_matrix_is_valid(input: &Vec<Vec<bool>>) -> bool {
        let len = input[0].len();
        for v in input {
            if v.len() != len {
                return false;
            }
        }

        return true;
    }

    /// Simulate the network using the provided input, which is a boolean matrix having
    /// dimension (nr. input neurons) x (nr. time steps). Each row corresponds to a particular
    /// entrance of the SNN, and each column corresponds to a certain time step.
    /// If input[i][j] == true, it means that, at time step 'j', the SNN receives a pulse on
    /// the entrance 'i'. Otherwise, if it false, no input is received for that time step.
    pub fn run(mut self, input: Vec<Vec<bool>>) -> Vec<Vec<bool>> {
        // Number of entrances of SNN, equal to the number of rows of the 'input' matrix
        let snn_inputs_number = input.len();

        // Number of time steps to simulate in the SNN, equal to the number of columns
        // of the 'input' matrix
        let snn_time_steps_number = input[0].len();

        //Time step duration converted to milliseconds to perform computation later
        let time_step_duration_ms = self.time_step_duration_us / 1000.0;

        // number of layers in the network
        let number_of_layers = self.layers.len();

        // create the boolean matrix used to hold the result of the simulation. The matrix has as many
        // rows as the number of output neurons and as many columns ad the number of time steps involved
        // in the simulation. If output[i][j] == true, it means that the exit 'i' produced a Pulse at time
        // step 'j'. Otherwise, if false, there is no pulse for that time step.
        //
        // The matrix is initialized with 'false' values.
        let mut output = self.create_output_matrix(snn_time_steps_number);

        // Each layer of the SNN is run in a separated thread. Different layers are able to
        // communicate, i.e. exchange Pulses or Control Messages, using channels; each layer
        // has access to:
        // - the receiver side of a channel to read messages from previous layer
        // - the sender side of another channel to send messages to the following layer
        //
        // Example:
        // {input} [tx_0]--->[rx_0] {layer 0} [tx_1]--->[rx_1] {layer 1} [tx_2]--->[rx_2] ...
        //
        // Usage of each channel side:
        // - input_injection_sender: it is used only once, to transfer the Pulses contained
        //      in the input matrix to layer 0 of the SNN, ordered by time step (all
        //      pulses sent for a certain time step are followed by a Control Message - GoAhead-
        //      used to signal that there are no more Pulses for that time step, so that the
        //      listener can stop waiting for messages on the channel and start its computation
        //      before starting waiting for Pulses belonging to the next time step).
        // - receiver_from_previous_layer: it represents the channel side used by each layer
        //      to receive Messages from the previous layer. Once the thread for a specific
        //      layer is spawned, this variable is updated with the content of
        //      'future_receiver_from_previous_layer' and fed again to the next thread.
        // - sender_to_following_layer: it is used by a layer to send Messages to the following
        //      layer. Once the thread is spawned, this variable is updated with the sender side of
        //      a newly created channel
        // - future_receiver_from_previous_layer: it represents the receiver coupled with the
        //      sender used to transfer messages to the following layer, so, once the thread is
        //      spawned, this receiver must be assigned to the variable 'receiver_from_previous_layer',
        //      so that it becomes the input channel for the next layer. Then its content is updated with
        //      the receiver of a newly created channel

        let (input_injection_sender, mut receiver_from_previous_layer) = mpsc::channel();
        let (mut sender_to_following_layer, mut future_receiver_from_previous_layer) =
            mpsc::channel();

        // Injecting Pulses from input matrix to layer 0, ordered by time step and separated
        // by using a GoAhead control message
        for time_step in 0..snn_time_steps_number {
            for input_node in 0..snn_inputs_number {
                if input[input_node][time_step] {
                    input_injection_sender
                        .send(Message::Pulse(input_node))
                        .unwrap();
                }
            }
            input_injection_sender.send(Message::GoAhead).unwrap();
        }

        // Create a Vec to hold thread handles
        let mut thread_handles = Vec::<JoinHandle<()>>::new();

        // Spawning a thread for each layer
        for layer_nr in 0..number_of_layers {
            // Each thread takes possession of the Vec containing the Neurons
            // for the corresponding layer
            let mut layer_neurons = self.layers.remove(0);

            let join_handle = thread::Builder::new()
                .name(format!("layer {}", layer_nr))
                .spawn(move || {
                    // Vec to keep track of all neurons of the current layer that had emitted pulses
                    // during the previous time step. This is needed to apply inhibitive contribution
                    // to the membrane potential of each neuron, so that the right internal weights
                    // can be selected for computation.
                    let mut emitted_pulse_sources = Vec::new();

                    // each layer operates one time step at a time, in order. In order to perform
                    // computation for time step 'k', it is necessary that the layer has received all
                    // Pulses emitted during the SAME time step by the previous layer.
                    // In other words, layer 'n' can only process time step 'k' data when layer 'n-1'
                    // has terminated its own computation on the same time step 'k'
                    for time_step in 0..snn_time_steps_number {
                        // Vec to keep track of the origin of each Pulse received during the current time step,
                        // i.e. the index of the neuron belonging to the previous layer - or entrance - which generated
                        // the Pulse itself; this is needed to allow the Neurons to choose the right Weight when
                        // computing the new Membrane Potential
                        let mut pulse_sources = Vec::new();

                        // Receive all pulses for the current time step
                        while let Ok(Message::Pulse(source)) = receiver_from_previous_layer.recv() {
                            pulse_sources.push(source);
                        }

                        // apply inhibitive contribution due to pulses generated by nodes of the current layer
                        // during previous time step (except for time step 0)
                        if time_step > 0 {
                            for neuron in layer_neurons.iter_mut() {
                                neuron.inhibite_after_pulses_emission(
                                    &emitted_pulse_sources,
                                    time_step,
                                    time_step_duration_ms,
                                    self.model,
                                );
                            }
                            // all neurons updated: clear emitted_pulse_sources Vec so that it can
                            // be used again, for next iteration
                            emitted_pulse_sources.clear();
                        }

                        // Update the status for the layer Neurons ONLY if at least a pulse
                        // is received by the layer, otherwise there is no need to do that.
                        if pulse_sources.len() > 0 {
                            // Feed Pulses to all neurons in the layer
                            for (i, neuron) in layer_neurons.iter_mut().enumerate() {
                                // if the current neuron 'fires', send a Pulse over the channel
                                // to the following layer
                                if neuron.feed_pulses(
                                    &pulse_sources,
                                    time_step,
                                    time_step_duration_ms,
                                    self.model,
                                ) {
                                    // add current neuron to emitted_pulse_sources
                                    emitted_pulse_sources.push(i);
                                    // send pulses over the channel
                                    sender_to_following_layer.send(Message::Pulse(i)).unwrap();
                                }
                            }

                            // Signal to the following layer that all pulses for this time step
                            // have been sent, by sending a GoAhead Control Message
                            sender_to_following_layer.send(Message::GoAhead).unwrap();
                        }
                    }
                });

            // Push the join handle inside the Vec
            thread_handles.push(join_handle.unwrap());

            // prepare the channels for the following thread (layer) to spawn

            // receiver_from_previous_layer is assigned the receiver coupled with the transmitter
            // of the previous thread, so that it can be used by the next thread that will be spwaned
            // to read Messages produced by the previous thread
            receiver_from_previous_layer = future_receiver_from_previous_layer;

            // The next thread that will be spawned has been given the receiver side, but it still does
            // not have a transmitter.

            // If the last thread has already been spawned, there is no next thread, so there is no need
            // to create another transmitter
            if layer_nr == number_of_layers - 1 {
                break;
            }

            // Otherwise, a new channel is created, whose transmitter is given to the next thread
            (
                sender_to_following_layer,
                future_receiver_from_previous_layer,
            ) = mpsc::channel();
        }

        // Await termination of all spawned threads
        thread_handles.into_iter().for_each(|join_handle| {
            join_handle.join().unwrap();
        });

        // write results to the output boolean matrix
        Self::write_results(receiver_from_previous_layer, &mut output);

        return output;
    }

    /// Simulate the behaviour of the SNN in presence of damages to its fundamental elements.
    /// The simulation is first run without applying any damages and then it is repeated
    /// as many times as specified by the 'iterations' parameter, applying the requested
    /// type of damage to only ONE random element whose type is chosen among those specified in
    /// the 'faulty_elements' parameter.
    /// 'input' boolean matrix is used to feed the desired input to the SNN.
    pub fn simulate(
        &self,
        faulty_elements: Vec<FaultyElement>,
        damage_type: DamageModel,
        iterations: usize,
        input: Vec<Vec<bool>>,
    ) -> Option<SimulationResult> {
        // check whether the input matrix has valid dimensions
        if !Self::input_matrix_is_valid(&input) {
            return None;
        }

        // create Simulation Result matrix
        let mut simulation_result_matrix = Vec::new();
        for i in 0..self.nr_outputs {
            simulation_result_matrix.push(Vec::new());
            for j in 0..input[0].len() {
                simulation_result_matrix[i].push(SimulationResultCell::new(i, j));
            }
        }

        // run the simulation without applying any damages to network elements
        let output_without_damages = self.clone().run(input.clone());

        // run the simulation as many times as specified by 'iterations' parameter, applying the
        // the chosen DamageModel ('damage_type') each time to a different element chosen randomly among
        // those specified in the 'faulty_elements' Vec.
        for iteration_number in 0..iterations {
            // clone the network, so that each instance can be Damaged independently
            let mut snn = self.clone();
            // apply damage to the snn
            let damage_detail =
                Self::apply_damage_to_snn(&mut snn, damage_type, &faulty_elements, input[0].len())
                    .unwrap();

            let output_with_damage = Self::run(snn, input.clone());

            // compare matrix to the one obtained without damages, updating result matrix
            Self::compare_outputs(
                &output_without_damages,
                &output_with_damage,
                &mut simulation_result_matrix,
                iteration_number,
                damage_detail,
            );
        }

        // filling result with actual values from simulation with damages for each output
        // and time step
        for i in 0..output_without_damages.len() {
            for j in 0..output_without_damages[0].len() {
                if simulation_result_matrix[i][j].diff_count != 0 {
                    simulation_result_matrix[i][j].actual_value = !output_without_damages[i][j];
                } else {
                    simulation_result_matrix[i][j].actual_value = output_without_damages[i][j];
                }
            }
        }

        // return result structure
        return Some(SimulationResult {
            number_of_iterations: iterations,
            type_of_damage: damage_type,
            output_without_damages,
            diffs: simulation_result_matrix,
        });
    }

    fn compare_outputs(
        output_without_damages: &Vec<Vec<bool>>,
        output_with_damage: &Vec<Vec<bool>>,
        simulation_result_matrix: &mut Vec<Vec<SimulationResultCell>>,
        iteration_number: usize,
        mut damage_detail: DamageDetail,
    ) {
        // compare the two output matrixes and update the result matrix
        for i in 0..output_with_damage.len() {
            for j in 0..output_with_damage[0].len() {
                if output_with_damage[i][j] != output_without_damages[i][j] {
                    damage_detail.at_iteration = iteration_number;
                    simulation_result_matrix[i][j].diff_count += 1;
                    simulation_result_matrix[i][j]
                        .damage_details
                        .push(damage_detail);
                }
            }
        }
    }

    /// Apply a single-bit Damage to the network: one element is chosen randomly among those
    /// listed in 'faulty_elements', and the specified DamageModel is applied.
    fn apply_damage_to_snn(
        &mut self,
        damage_type: DamageModel,
        faulty_elements: &Vec<FaultyElement>,
        number_of_time_steps: usize,
    ) -> Option<DamageDetail> {
        // create a random number generator
        let mut rng = thread_rng();

        // choose a random element
        match faulty_elements.choose(&mut rng) {
            // if an element is found
            Some(faulty_element) => {
                // choose a random layer
                let index_of_layer_to_damage = rand::thread_rng().gen_range(0..self.layers.len());
                let layer_to_damage = &mut self.layers[index_of_layer_to_damage];
                // choose a random neuron
                let index_of_neuron_to_damage =
                    rand::thread_rng().gen_range(0..layer_to_damage.len());
                let neuron_to_damage = &mut layer_to_damage[index_of_neuron_to_damage];
                // choose bit position where to apply the damage (between 0 and 63 - since
                // Registers are on 64 bits)
                let bit_position = rng.gen_range(0..64) as usize;
                // choose time step when to apply the damage. This is needed for Damage models
                // which only affect the Register behaviour during precise time instants.
                let time_step = rng.gen_range(0..number_of_time_steps);
                // create damage object
                let damage = match damage_type {
                    DamageModel::StuckAt0 => Damage::StuckAt0 { bit_position },
                    DamageModel::StuckAt1 => Damage::StuckAt1 { bit_position },
                    DamageModel::TransientBitFlip => Damage::TransientBitFlip {
                        bit_position,
                        time_step,
                    },
                };

                // apply damage to the correct Register
                match faulty_element {
                    FaultyElement::Weights => {
                        // choose randomly whether to damage external or internal weights
                        let weights_vec = if rng.gen_bool(0.5) {
                            &mut neuron_to_damage.weights
                        } else {
                            &mut neuron_to_damage.internal_weights
                        };
                        // choose randomly a weight to damage
                        let weight = weights_vec.choose_mut(&mut rng).unwrap();
                        // apply damage to the Register containing the weight
                        weight.apply_damage(damage);
                    }
                    FaultyElement::Thresholds => {
                        // apply damage to the Register containing v_th
                        neuron_to_damage.v_th.apply_damage(damage);
                    }
                    FaultyElement::MembranePotentials => {
                        // apply damage to the Register containing v_mem
                        neuron_to_damage.v_mem.apply_damage(damage);
                    }
                    FaultyElement::ResetPotentials => {
                        // apply damage to the Register containing v_reset
                        neuron_to_damage.v_reset.apply_damage(damage);
                    }
                    FaultyElement::PotentialsAtRest => {
                        // apply damage to the Register containing v_rest
                        neuron_to_damage.v_rest.apply_damage(damage);
                    }
                    FaultyElement::Comparator => {
                        neuron_to_damage.cmp_reg.apply_damage(damage);
                    }
                    FaultyElement::Adder => {
                        neuron_to_damage.add_reg.apply_damage(damage);
                    }
                    FaultyElement::Multiplier => {
                        neuron_to_damage.mul_reg.apply_damage(damage);
                    }
                    FaultyElement::Divider => {
                        neuron_to_damage.div_reg.apply_damage(damage);
                    }
                }

                // struct which describes the damage in detail. The field 'at_iteration' here is dummy,
                // since it will be replaced if and when a difference between the expected output and
                // actual one is found.
                return Some(DamageDetail {
                    at_iteration: 0,
                    damage_type: *faulty_element,
                    at_layer: index_of_layer_to_damage,
                    at_neuron: index_of_neuron_to_damage,
                    at_bit: bit_position,
                });
            }
            // if no element is found, return
            None => {
                return None;
            }
        }
    }
}
