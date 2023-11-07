use rand::Rng;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, BitAndAssign, BitOrAssign, Mul, Not};
use std::rc::Rc;

use super::{LogicCircuit, ToBits};

#[derive(Clone)]
pub enum Stuck {
    Zero,
    One,
    Transient,
}
#[derive(Debug, Clone)]
pub struct FullAdderTree<T, U>
where
    T: Clone,
{
    root: Node<T>,

    marker: PhantomData<U>,
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
pub struct FullAdder<T: Clone> {
    input1: T,
    input2: T,
    output: T,
    error_selector: Option<(u8, u64)>,
}

impl<T, U> FullAdderTree<T, U>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U> + Default + Debug + Display,
    U: BitOrAssign + Not<Output = U> + BitAndAssign + Clone,
{
    pub fn new(num_inputs: usize) -> FullAdderTree<T, U> {
        let random_index = rand::thread_rng().gen_range(0..num_inputs - 1); //select index of the full adder
        let field: u8 = rand::thread_rng().gen_range(1..=3); //selct field i1,i2,outpur
        let bit: u64 = rand::thread_rng().gen_range(0..64); //sleect bit

        let error_selector = (field, bit);
        let mut current_index = 0;
        let root = FullAdderTree::create_tree(
            num_inputs,
            &mut current_index,
            random_index,
            error_selector,
        );
        FullAdderTree {
            root: root,
            marker: PhantomData,
        }
    }

    fn create_tree(
        num_inputs: usize,
        curr_index: &mut usize,
        random_index: usize,
        error_selector: (u8, u64),
    ) -> Node<T> {
        if num_inputs == 2 {
            let mut full_adder = FullAdder::new(T::default(), T::default());
            if *curr_index == random_index {
                full_adder.set_error_selector(Some(error_selector));
            }
            *curr_index += 1;
            let adder = Node::FullAdderNode(FullAdderNode {
                full_adder: full_adder.clone(),
                left: Some(Box::new(Node::Value(T::default()))),
                right: Some(Box::new(Node::Value(T::default()))),
            });

            return adder;
        } else if num_inputs < 2 {
            let value = Node::Value(T::default());
            return value;
        } else {
            let half = (num_inputs + 1) / 2;

            let left_tree =
                FullAdderTree::create_tree(half, curr_index, random_index, error_selector);
            let right_tree = FullAdderTree::create_tree(
                num_inputs - half,
                curr_index,
                random_index,
                error_selector,
            );

            let mut sum_tree = FullAdder::new(T::default(), T::default());

            if *curr_index == random_index {
                sum_tree.set_error_selector(Some(error_selector));
            }
            *curr_index += 1;

            let adder = Node::FullAdderNode(FullAdderNode {
                full_adder: sum_tree.clone(),
                left: Some(Box::new(left_tree)),
                right: Some(Box::new(right_tree)),
            });
            return adder;
        }
    }

    pub fn solve_tree_addition(&mut self, inputs: Rc<Vec<T>>, stuck: Stuck) -> T {
        FullAdderTree::solve_recursive(&mut self.root, inputs, stuck, &mut 0)
    }

    fn solve_recursive(
        node: &mut Node<T>,
        inputs: Rc<Vec<T>>,
        stuck: Stuck,
        index: &mut usize,
    ) -> T {
        match node {
            Node::FullAdderNode(full_adder_node) => {
                if let Some(left) = &mut full_adder_node.left {
                    let value = Self::solve_recursive(left, Rc::clone(&inputs), stuck.clone(), index);
                    full_adder_node.full_adder.set_input1(value.clone());
                }
                if let Some(right) = &mut full_adder_node.right {
                    let value = Self::solve_recursive(right, Rc::clone(&inputs), stuck.clone(), index);
                    full_adder_node.full_adder.set_input2(value.clone());
                }

                //compute sum with method in full_adder
                let sum = full_adder_node.full_adder.operation(stuck.clone());

                full_adder_node.full_adder.set_output(sum.clone());

                return sum;
            }
            Node::Value(_value) => {
                if *index < inputs.len() {
                    let value = &inputs[*index];
                    *index += 1;
                    value.clone()
                } else {
                    panic!("Index out of bounds");
                }
            }
        }
    }
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
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U> + Display,
    U: BitOrAssign + Not<Output = U> + BitAndAssign + Clone,
{
    fn operation(&mut self, stuck: Stuck) -> T {
        println!("i1:{}  i2:{} ", self.get_input1(), self.get_input2());
        if self.error_selector.is_some() {
            println!(
                "error_selector: {} {}",
                self.error_selector.unwrap().0,
                self.error_selector.unwrap().1
            );
            self.set_random_bit(self.error_selector.unwrap(), stuck);
            println!(
                "i1:{}  i2:{} output:{} ",
                self.get_input1(),
                self.get_input2(),
                self.get_output()
            );
            return self.get_output();
        }

        self.input1.clone() + self.input2.clone()
    }

    fn set_random_bit(&mut self, indexes: (u8, u64), stuck: Stuck) {
        set_random_field(self, indexes, stuck);
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
    fn get_error_selector(&self) -> Option<(u8, u64)> {
        self.error_selector
    }

    // Setter per error_selector
    fn set_error_selector(&mut self, value: Option<(u8, u64)>) {
        self.error_selector = value;
    }
}

#[derive(Debug, Clone)]
pub struct Multiplier<T: Clone> {
    input1: T,
    input2: T,
    output: T,
    error_selector: Option<(u8, u64)>,
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
    fn operation(&mut self, stuck:Stuck) -> T {
        self.input1.clone() * self.input2.clone()
    }

    fn set_random_bit(&mut self, indexes: (u8, u64), stuck: Stuck) {
        //set_random_field(self, stuck);
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

    fn get_error_selector(&self) -> Option<(u8, u64)> {
        self.error_selector
    }

    // Setter per error_selector
    fn set_error_selector(&mut self, value: Option<(u8, u64)>) {
        self.error_selector = value;
    }
}

/*
It sets a random bit in the circuit's inputs or output based on the stuck flag.
The function determines which input/output to modify based on the error selector (if available) or a random choice.
It applies the error injection and set the new value to do the operations */
fn set_random_field<T, U>(circuit: &mut dyn LogicCircuit<T, U>, indexes: (u8, u64), stuck: Stuck)
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U>,
    U: BitOrAssign + Not<Output = U> + BitAndAssign,
{
    //select randomly the field to modify if not already selected
    let (field_to_set, bit_index) = indexes;

    match field_to_set {
        1 => {
            let value = circuit.get_input1();
            let new_val = apply_injection(value, stuck, bit_index);
            circuit.set_input1(new_val);
            circuit.set_output(circuit.get_input1() + circuit.get_input2());
        }
        2 => {
            let value = circuit.get_input2();
            let new_val = apply_injection(value, stuck, bit_index);
            circuit.set_input2(new_val);
            circuit.set_output(circuit.get_input1() + circuit.get_input2());
        }
        3 => {
            let value = circuit.get_input1() + circuit.get_input2();
            let new_val = apply_injection(value, stuck, bit_index);
            circuit.set_output(new_val);
        }
        _ => panic!("Invalid field selected"),
    };
}

/*
It modifies the value by either setting or clearing a specific bit based on the stuck flag and the provided error selector.
It returns the modified value. */
pub fn apply_injection<T, U>(value: T, stuck: Stuck, index: u64) -> T
where
    T: ToBits<U>,
    U: BitOrAssign + Not<Output = U> + BitAndAssign,
{
    let mut bits: U = value.get_bits(); //get the bit rapresentation of the value

    match stuck {
        Stuck::Zero => bits &= !(value.create_mask(index)),
        Stuck::One => bits |= value.create_mask(index),
        Stuck::Transient => {}
    };

    T::from_bits(bits)
}
