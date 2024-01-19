/* The register module defines:
- an enum Damage which represents the kind of damage which could occour
inside a register
- a struct Register which represents a model of an hardware register containing
floating point values on 64 bits (f64 values). */

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

    /// write the provided value to the register.
    /// 'Damages', if present, are applied each time ONLY to the returned copy of the
    /// value when performing a reading, so, leaving the original unchanged
    pub fn write_value(&mut self, value: f64) {
        self.value = value;
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
        let mut int_val: u64 = unsafe { std::mem::transmute(value) };
        /* Apply mask and */
        int_val &= mask;

        /* Convert u64 back into f64 */
        let res: f64 = unsafe { std::mem::transmute(int_val) };

        /* Return res */
        res
    }

    fn bitwise_or(value: f64, mask: u64) -> f64 {
        /* Convert f64 into a u64 */
        let mut int_val: u64 = unsafe { std::mem::transmute(value) };
        /* Apply mask and */
        int_val |= mask;

        /* Convert u64 back into f64 */
        let res: f64 = unsafe { std::mem::transmute(int_val) };

        /* Return res */
        res
    }

    fn bitwise_xor(value: f64, mask: u64) -> f64 {
        /* Convert f64 into a u64 */
        let mut int_val: u64 = unsafe { std::mem::transmute(value) };
        /* Apply mask and */
        int_val ^= mask;

        /* Convert u64 back into f64 */
        let res: f64 = unsafe { std::mem::transmute(int_val) };

        /* Return res */
        res
    }

    pub fn cmp(r1: Self, r2: Self, res_reg: &mut Self, current_time_step: usize) {
        // reading content of r1 and r2
        let n1 = r1.read_value(Some(current_time_step)).unwrap();
        let n2 = r2.read_value(Some(current_time_step)).unwrap();

        // computing result
        let res = n1 - n2;

        // storing result
        res_reg.write_value(res);
    }

    pub fn add(r1: Self, r2: Self, res_reg: &mut Self, current_time_step: usize) {
        // reading content of r1 and r2
        let n1 = r1.read_value(Some(current_time_step)).unwrap();
        let n2 = r2.read_value(Some(current_time_step)).unwrap();

        // computing result
        let res = n1 + n2;

        // storing result
        res_reg.write_value(res);
    }

    pub fn sub(r1: Self, r2: Self, res_reg: &mut Self, current_time_step: usize) {
        // reading content of r1 and r2
        let n1 = r1.read_value(Some(current_time_step)).unwrap();
        let n2 = r2.read_value(Some(current_time_step)).unwrap();

        // computing result
        let res = n1 - n2;

        // storing result
        res_reg.write_value(res);
    }

    pub fn mult(r1: Self, r2: Self, res_reg: &mut Self, current_time_step: usize) {
        // reading content of r1 and r2
        let n1 = r1.read_value(Some(current_time_step)).unwrap();
        let n2 = r2.read_value(Some(current_time_step)).unwrap();

        // computing result
        let res = n1 * n2;

        // storing result
        res_reg.write_value(res);
    }

    pub fn div(r1: Self, r2: Self, res_reg: &mut Self, current_time_step: usize) {
        // reading content of r1 and r2
        let n1 = r1.read_value(Some(current_time_step)).unwrap();
        let n2 = r2.read_value(Some(current_time_step)).unwrap();

        // computing result
        let res = n1 / n2;

        // storing result
        res_reg.write_value(res);
    }

    pub fn copy_to(&self, dest_reg: &mut Self, current_time_step: usize) {
        dest_reg.write_value(self.read_value(Some(current_time_step)).unwrap());
    }
}
