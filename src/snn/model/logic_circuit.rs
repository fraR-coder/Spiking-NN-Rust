use rand::Rng;
use std::ops::{Add, BitAndAssign, BitOrAssign, Mul, Not};

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
pub trait LogicCircuit<T: Add<Output = T> + Mul<Output = T> + Clone, U> {
    fn operation(&mut self, stuck: bool) -> T;
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

#[derive(Debug, Clone)]
pub enum Node<T: Clone> {
    FullAdderNode(FullAdderNode<T>),
    Value(T),
}
#[derive(Debug, Clone)]
pub struct FullAdderNode<T: Clone> {
    full_adder: FullAdder<T>,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
}

#[derive(Debug, Clone)]
pub struct FullAdderTree<T: Clone> {
    root: Option<Node<T>>,
    full_adders_list: Vec<FullAdder<T>>,
}

impl<T: Clone + Default> FullAdderTree<T> {
    pub fn new(num_inputs: usize) -> FullAdderTree<T> {
        let (root, full_adders_list) = FullAdderTree::create_tree_and_list(num_inputs);
        FullAdderTree {
            root: Some(root),
            full_adders_list,
        }
    }

    fn create_tree_and_list(num_inputs: usize) -> (Node<T>, Vec<FullAdder<T>>) {
        if num_inputs == 2 {
            let full_adder = FullAdder::new(T::default(), T::default());
            let adder = Node::FullAdderNode(FullAdderNode {
                full_adder: full_adder.clone(),
                left: None,
                right: None,
            });

            return (adder, vec![full_adder]);
        } else if num_inputs < 2 {
            let value = Node::Value(T::default());
            return (value, vec![]);
        } else {
            let half = num_inputs / 2;
            let (left_tree, left_full_adders) = FullAdderTree::create_tree_and_list(half);
            let (right_tree, right_full_adders) =
                FullAdderTree::create_tree_and_list(num_inputs - half);

            let sum_tree = FullAdder::new(T::default(), T::default());
            let mut full_adders = vec![sum_tree.clone()];
            full_adders.extend(left_full_adders);
            full_adders.extend(right_full_adders);

            let adder = Node::FullAdderNode(FullAdderNode {
                full_adder: sum_tree.clone(),
                left: None,
                right: None,
            });
            return (adder, full_adders);
        }
    }

    pub fn get_num_full_adders(&self) -> usize {
        self.full_adders_list.len()
    }

    pub fn get_full_adder_mut(&mut self, index: usize) -> Option<&mut FullAdder<T>> {
        self.full_adders_list.get_mut(index)
    }

    pub fn solve_tree_addition(&mut self, inputs: Vec<T>) {
        if let Some(root) = &self.root {
            FullAdderTree::solve_recursive(root, inputs);
        };
    }

    fn solve_recursive(node: &Node<T>, inputs: Vec<T>)->&T {
        match node {
            Node::FullAdderNode(full_adder_node) => {
                if let Some(left)=&full_adder_node.left{
                    let value=Self::solve_recursive(left, inputs);
                    full_adder_node.full_adder.input1=value.clone();
                }
                if let Some(right)=&full_adder_node.right{
                    let value=Self::solve_recursive(right, inputs);
                    full_adder_node.full_adder.input2=value.clone();
                }

                //compute sum with method in full_adder
                let sum=&T::default();
                return sum;

            }
            Node::Value(value)=>{
                return value;
            }
        };
    }
}

#[derive(Debug, Clone)]
pub struct FullAdder<T: Clone> {
    input1: T,
    input2: T,
    output: T,
    error_selector: Option<(i32, i32)>,
}

impl<T> FullAdder<T>
where
    T: Clone + Default,
{
    pub fn new(input_left: T, input_right: T) -> FullAdder<T> {
        return Self {
            input1: input_left,
            input2: input_right,
            error_selector: None,
            output: T::default(),
        };
    }
}

impl<T, U> LogicCircuit<T, U> for FullAdder<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U>,
    U: BitOrAssign + Not<Output = U> + BitAndAssign + Clone,
{
    fn operation(&mut self, stuck: bool) -> T {
        self.set_random_bit(stuck);
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

#[derive(Debug, Clone)]
pub struct Multiplier<T: Clone> {
    input1: T,
    input2: T,
    output: T,
    error_selector: Option<(i32, i32)>,
}

impl<T> Multiplier<T>
where
    T: Clone + Default,
{
    pub fn new() -> Multiplier<T> {
        return Self {
            input1: T::default(),
            input2: T::default(),
            error_selector: None,
            output: T::default(),
        };
    }
}

impl<T, U> LogicCircuit<T, U> for Multiplier<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U>,
    U: BitOrAssign + Not<Output = U> + BitAndAssign,
{
    fn operation(&mut self, stuck: bool) -> T {
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
    let mut error_selector = circuit.get_error_selector();

    //select randomly the field to modify if not already selected
    let field_to_set = match error_selector {
        Some(index) => index.0,
        None => rand::thread_rng().gen_range(1..=3),
    };

    match field_to_set {
        1 => {
            let value = circuit.get_input1();
            let new_val = apply_injection(value, stuck, &mut error_selector, field_to_set);
            circuit.set_input1(new_val);
        }
        2 => {
            let value = circuit.get_input2();
            let new_val = apply_injection(value, stuck, &mut error_selector, field_to_set);
            circuit.set_input2(new_val);
        }
        3 => {
            let value = circuit.get_output();
            let new_val = apply_injection(value, stuck, &mut error_selector, field_to_set);
            circuit.set_output(new_val);
        }
        _ => panic!("Invalid field selected"),
    };
}

/*
It modifies the value by either setting or clearing a specific bit based on the stuck flag and the provided error selector.
It returns the modified value. */
pub fn apply_injection<T, U>(value: T, stuck: bool, index: &mut Option<(i32, i32)>, field: i32) -> T
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
            //update value of index for the error for the enxt iterations

            *index = Some((field, rand::thread_rng().gen_range(0..num_bits)));

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
