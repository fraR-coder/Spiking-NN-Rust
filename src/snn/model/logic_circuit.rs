use rand::Rng;
use std::{ops::{Add, BitAndAssign, BitOrAssign, Mul, Not}, rc::Rc};

// Define the ToBits trait
pub trait ToBits<U> {
    fn get_bits(&self) -> U;
    fn from_bits(bits: U) -> Self;

    fn create_mask(&self, index: i32) -> U; //just to create a mask of bits for different types
                                            //generate a bit sequence with all 0's and a 1 in the position specified by index.
                                            //The lenght depends on the type implementing this

    // Get the number of bits in the type implementing this trait.
    fn num_bits(&self) -> i32;
}

impl ToBits<u64> for f64 {
    // to conversion with bits
    fn get_bits(&self) -> u64 {
        self.to_bits()
    }
    fn from_bits(bits: u64) -> Self {
        f64::from_bits(bits)
    }

    fn create_mask(&self, index: i32) -> u64 {
        1u64 << index
    }

    fn num_bits(&self) -> i32 {
        64 //Modo migliore????
    }
}
//**TODO can be implemented for all the types we need */
/*
The LogicCircuit trait represents a generic logic circuit that performs operations and provides methods
for setting and getting inputs, outputs, and error selectors.
It is parameterized with two types, T and U. T is the type of the values taht perform the operations. U is the type of the rappresentation in bit of T (e.g. T:f64->U:u64)
The error selector is a field to keep track of the selected field( i1,i2,o) and selected bit to apply the error bit injection
*/
pub trait LogicCircuit<T: Add<Output = T> + Mul<Output = T>+Clone, U> {
    fn operation(&self) -> T;
    fn set_random_bit(&mut self, stuck: bool);
    fn get_input1(&self) -> T;
    fn set_input1(&mut self, value: T);
    fn get_input2(&self) -> T;
    fn set_input2(&mut self, value: T);
    fn get_output(&self) -> T;
    fn set_output(&mut self, value: T);
    fn get_error_selector(&self) -> Option<(i32, i32)>;
    fn set_error_selector(&mut self, value: Option<(i32, i32)>);
}

#[derive(Debug,Clone)]
pub struct FullAdderTree<T: Clone> {
    full_adder: Option<FullAdder<T>>,
    children: Option<Vec<Box<FullAdderTree<T>>>>,
}

impl<T: Clone + Default> FullAdderTree<T> {
    pub fn new(num_inputs: usize) -> FullAdderTree<T> {
        if num_inputs <= 2 {
            return FullAdderTree {
                full_adder: Some(FullAdder::new(T::default(), T::default())),
                children: None,
            };
        } else {
            let half = num_inputs / 2;
            let left_tree:Box<FullAdderTree<T>> = Box::new(FullAdderTree::new(half));
            let right_tree:Box<FullAdderTree<T>> = Box::new(FullAdderTree::new(num_inputs - half));

            let mut sum_tree = FullAdder::new(T::default(), T::default());
            

            FullAdderTree {
                full_adder: Some(sum_tree),
                children: Some(vec![left_tree, right_tree]),
            }
        }
    }

    
}

#[derive(Debug,Clone)]
pub struct FullAdder<T:Clone> {
    input1: T,
    input2: T,
    output: T,
    error_selector: Option<(i32, i32)>,
}

impl<T> FullAdder<T> where T:Clone+ Default {
    pub fn new(input_left:T, input_right:T) -> FullAdder<T> {
        return Self {
            input1:input_left,
            input2:input_right,
            error_selector:None,
            output: T::default(),
            
        };
    }
}

impl<T, U> LogicCircuit<T, U> for FullAdder<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U>,
    U: BitOrAssign + Not<Output = U> + BitAndAssign+ Clone,
{
    fn operation(&self) -> T {
        self.input1.clone() + self.input2.clone()
    }

    fn set_random_bit(&mut self, stuck: bool) {
        set_random_field(self, stuck);
    }

    fn get_input1(&self) -> T {
        self.input1.clone()
    }

    fn set_input1(&mut self, value: T) {
        self.input1 = value;
    }

    fn get_input2(&self) -> T {
        self.input2.clone()
    }

    fn set_input2(&mut self, value: T) {
        self.input2 = value;
    }

    fn get_output(&self) -> T {
        self.output.clone()
    }

    fn set_output(&mut self, value: T) {
        self.output = value;
    }

    // Getter per error_selector
    fn get_error_selector(&self) -> Option<(i32, i32)> {
        self.error_selector
    }

    // Setter per error_selector
    fn set_error_selector(&mut self, value: Option<(i32, i32)>) {
        self.error_selector = value;
    }
}

#[derive(Debug,Clone)]
pub struct Multiplier<T:Clone> {
    input1: T,
    input2: T,
    output: T,
    error_selector: Option<(i32, i32)>,
}

impl<T> Multiplier<T> where T:Clone+ Default {
    pub fn new() -> Multiplier<T> {
        return Self {
            input1:T::default(),
            input2:T::default(),
            error_selector:None,
            output: T::default(),
            
        };
    }
}

impl<T, U> LogicCircuit<T, U> for Multiplier<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U>,
    U: BitOrAssign + Not<Output = U> + BitAndAssign,
{
    fn operation(&self) -> T {
        self.input1.clone() * self.input2.clone()
    }

    fn set_random_bit(&mut self, stuck: bool) {
        set_random_field(self, stuck);
    }

    fn get_input1(&self) -> T {
        self.input1.clone()
    }

    fn set_input1(&mut self, value: T) {
        self.input1 = value;
    }

    fn get_input2(&self) -> T {
        self.input2.clone()
    }

    fn set_input2(&mut self, value: T) {
        self.input2 = value;
    }

    fn get_output(&self) -> T {
        self.output.clone()
    }

    fn set_output(&mut self, value: T) {
        self.output = value;
    }

    fn get_error_selector(&self) -> Option<(i32, i32)> {
        self.error_selector
    }

    // Setter per error_selector
    fn set_error_selector(&mut self, value: Option<(i32, i32)>) {
        self.error_selector = value;
    }
}

/*
It sets a random bit in the circuit's inputs or output based on the stuck flag.
The function determines which input/output to modify based on the error selector (if available) or a random choice.
It applies the error injection and set the new value to do the operations */
fn set_random_field<T, U>(circuit: &mut dyn LogicCircuit<T, U>, stuck: bool)
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U>,
    U: BitOrAssign + Not<Output = U> + BitAndAssign,
{
    let error_selector = circuit.get_error_selector();

    //select randomly the field to modify if not already selected
    let field_to_set = match error_selector {
        Some(index) => index.0,
        None => rand::thread_rng().gen_range(1..=3),
    };

    match field_to_set {
        1 => {
            let value = circuit.get_input1();
            let new_val = apply_injection(value, stuck, error_selector);
            circuit.set_input1(new_val);
        }
        2 => {
            let value = circuit.get_input2();
            let new_val = apply_injection(value, stuck, error_selector);
            circuit.set_input2(new_val);
        }
        3 => {
            let value = circuit.get_output();
            let new_val = apply_injection(value, stuck, error_selector);
            circuit.set_output(new_val);
        }
        _ => panic!("Invalid field selected"),
    };
}

/*
It modifies the value by either setting or clearing a specific bit based on the stuck flag and the provided error selector.
It returns the modified value. */
pub fn apply_injection<T, U>(value: T, stuck: bool, index: Option<(i32, i32)>) -> T
where
    T: ToBits<U>,
    U: BitOrAssign + Not<Output = U> + BitAndAssign,
{
    let mut bits: U = value.get_bits(); //get the bit rapresentation of the value

    //select randomly the bit to modify if not already selected
    let random_bit_index = match index {
        Some(bit_index) => bit_index.1,
        None => {
            let num_bits = value.num_bits();
            rand::thread_rng().gen_range(0..num_bits)
        }
    };

    if stuck {
        // if stuck_at_bit_1
        bits |= value.create_mask(random_bit_index);
    } else {
        bits &= !(value.create_mask(random_bit_index));
    }

    T::from_bits(bits)
}
