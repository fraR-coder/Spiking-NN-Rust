use rand::Rng;
use std::ops::{Add, Mul};

pub trait LogicCircuit<T: Add<Output = T> + Mul<Output = T>, U> {
    fn operation(&self) -> T;
    fn set_bit(&mut self, value: u8);
    fn get_input1(&mut self) -> &mut T;
    fn set_input1(&mut self, value: T);
    fn get_input2(&mut self) -> &mut T;
    fn set_input2(&mut self, value: T);
    fn get_output(&mut self) -> &mut T;
    fn set_output(&mut self, value: T);
}

pub struct FullAdder<T, U> {
    input1: T,
    input2: T,
    output: T,
    stuck_bit: U,
}

impl<T, U> LogicCircuit<T, U> for FullAdder<T, U>
where
    T: Add<Output = T> + Mul<Output = T> + Clone,
{
    fn operation(&self) -> T {
        self.input1.clone() + self.input2.clone()
    }

    fn set_bit(&mut self, value: u8) {
        set_random_field(self, value);
    }

    fn get_input1(&mut self) -> &mut T {
        &mut self.input1
    }

    fn set_input1(&mut self, value: T) {
        self.input1 = value;
    }

    fn get_input2(&mut self) -> &mut T {
        &mut self.input2
    }

    fn set_input2(&mut self, value: T) {
        self.input2 = value;
    }

    fn get_output(&mut self) -> &mut T {
        &mut self.output
    }

    fn set_output(&mut self, value: T) {
        self.output = value;
    }
}

pub struct Multiplier<T, U> {
    input1: T,
    input2: T,
    output: T,
    stuck_bit: U,
}

impl<T, U> LogicCircuit<T, U> for Multiplier<T, U>
where
    T: Add<Output = T> + Mul<Output = T> + Clone,
{
    fn operation(&self) -> T {
        self.input1.clone() * self.input2.clone()
    }

    fn set_bit(&mut self, value: u8) {
        set_random_field(self, value);
    }

    fn get_input1(&mut self) -> &mut T {
        &mut self.input1
    }

    fn set_input1(&mut self, value: T) {
        self.input1 = value;
    }

    fn get_input2(&mut self) -> &mut T {
        &mut self.input2
    }

    fn set_input2(&mut self, value: T) {
        self.input2 = value;
    }

    fn get_output(&mut self) -> &mut T {
        &mut self.output
    }

    fn set_output(&mut self, value: T) {
        self.output = value;
    }
}

fn set_random_field<T, U>(circuit: &mut dyn LogicCircuit<T, U>, value: u8)
where
    T: Add<Output = T> + Mul<Output = T> + Clone,
{
    let mut rng = rand::thread_rng();
    let field_to_set = rng.gen_range(1..=3); // Randomly select a field (1, 2, or 3)
    let field = match field_to_set {
        1 => circuit.get_input1(),
        2 => circuit.get_input2(),
        3 => circuit.get_output(),
        _ => panic!("Invalid field selected"),
    };

    //let bit = get_random_bit(/*field*/);
    let new_val = do_logic_operation(/*bit*/);

    *field = new_val;
}

fn get_random_bit<T>(/*field: &T*/) -> bool {
    // Implement logic to get a random bit based on the field's value
    // This is just a placeholder function
    // You need to implement your logic here.
    unimplemented!()
}

fn do_logic_operation<T>(/*bit: bool*/) -> T {
    // Implement logic to perform a logic operation based on the bit
    // This is just a placeholder function
    // You need to implement your logic here.
    unimplemented!()
}
