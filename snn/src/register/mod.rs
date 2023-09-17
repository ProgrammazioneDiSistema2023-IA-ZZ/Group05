/* The register module defines:
- an enum FaultyElement which lists the type of elements which can be potentially subject
to damages
- an enum Damage which represents the kind of damage which could occour
inside a register
- a struct Register which represents a model of an hardware register containing
floating point values on 64 bits (f64 values). */

#[derive(Clone, Copy, PartialEq)]
pub enum FaultyElement {
    Weights,
    Thresholds,
    MembranePotentials,
    ResetPotentials,
    PotentialsAtRest,
}

#[derive(Clone, Copy)]
pub enum Damage {
    /// the bit at the specified position is forced to 0 whenever the value
    /// is read or written from the register
    StuckAt0 { bit_position: usize },
    /// the bit at the specified position is forced to 1 whenever the value
    /// is read or written from the register
    StuckAt1 { bit_position: usize },
    /// the bit at the specified position is inverted when read ONLY at the
    /// specified time step (transient). This damage has no impact during
    /// other time steps
    TransientBitFlip {
        bit_position: usize,
        time_step: usize,
    },
    /// all the bits are working correctly
    Working,
}

#[derive(Clone, Copy)]
pub struct Register {
    value: f64,
    damage: Damage,
}

impl Register {
    /// initialize a new register with the provided value
    /// without inserting any damages
    pub fn new(value: f64) -> Self {
        Self {
            value,
            damage: Damage::Working,
        }
    }

    /// apply a damage to an existing register
    pub fn apply_damage(&mut self, damage: Damage) {
        self.damage = damage;
    }

    /// read and return the value contained inside the register: damages inside
    /// the register are automatically applied, if present.
    /// current_time_step can be set to None unless TransientBitFlip is
    /// used. If, in that case, None is passed ad current_time_step, the
    /// function returns None.
    pub fn read_value(&self, current_time_step: Option<usize>) -> Option<f64> {
        match self.damage {
            Damage::Working => {
                /* The value to be returned is not damaged, so it can
                be returned as it is */
                return Some(self.value);
            }
            Damage::StuckAt0 { bit_position } => {
                /* The value to be returned must have a 0 at the specified
                bit position */

                /* prepare a mask having all bits to 1, except for a 0 at position
                bit_position. The mask is then inverted bitwise, so that it is made up
                of all 1 except for a 0 at position bit_position */
                let mut mask = (1 as u64) << bit_position;
                mask = !mask;

                /* Apply the mask to the value and return */
                return Some(Self::bitwise_and(self.value, mask));
            }
            Damage::StuckAt1 { bit_position } => {
                /* The value to be returned must have a 1 at the specified
                bit position */

                /* prepare a mask having all 0, except for a 1 at position
                bit_position */
                let mask = (1 as u64) << bit_position;

                /* Apply the mask to the value and return */
                return Some(Self::bitwise_or(self.value, mask));
            }
            Damage::TransientBitFlip {
                bit_position,
                time_step,
            } => {
                /* TransientBitFlip is only applied at a specific time step.
                The value to be returned must have the bit at the specified position flipped. */

                /* If no time step is specified (current_time_step = None), then the function
                returns None. */
                if let None = current_time_step {
                    return None;
                }

                /* If the current_time_step differs from the one specified inside
                the TransientBitFlip, then the register value can be returned as it is */
                if let Some(curr_step) = current_time_step {
                    if curr_step != time_step {
                        return Some(self.value);
                    }
                }

                /* prepare a mask having all 0, except for a 1 at position
                bit_position */
                let mask = (1 as u64) << bit_position;

                /* Apply the mask to the value and return */
                return Some(Self::bitwise_xor(self.value, mask));
            }
        }
    }

    fn bitwise_and(value: f64, mask: u64) -> f64 {
        /* Convert f64 into a u64 */
        let mut int_val: u64 = 0;
        std::mem::swap(&mut int_val, unsafe { std::mem::transmute(value) });

        /* Apply mask and */
        int_val &= mask;

        /* Convert u64 back into f64 */
        let mut res: f64 = 0.0;
        std::mem::swap(&mut res, unsafe { std::mem::transmute(int_val) });

        /* Return res */
        res
    }

    fn bitwise_or(value: f64, mask: u64) -> f64 {
        /* Convert f64 into a u64 */
        let mut int_val: u64 = 0;
        std::mem::swap(&mut int_val, unsafe { std::mem::transmute(value) });

        /* Apply mask or */
        int_val |= mask;

        /* Convert u64 back into f64 */
        let mut res: f64 = 0.0;
        std::mem::swap(&mut res, unsafe { std::mem::transmute(int_val) });

        /* Return res */
        res
    }

    fn bitwise_xor(value: f64, mask: u64) -> f64 {
        /* Convert f64 into a u64 */
        let mut int_val: u64 = 0;
        std::mem::swap(&mut int_val, unsafe { std::mem::transmute(value) });

        /* Apply mask xor */
        int_val ^= mask;

        /* Convert u64 back to f64 */
        let mut res: f64 = 0.0;
        std::mem::swap(&mut res, unsafe { std::mem::transmute(int_val) });

        /* Return res */
        res
    }
}
