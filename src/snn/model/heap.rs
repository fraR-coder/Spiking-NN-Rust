//! # Heap Calculator
//!
//! This module defines a heap calculator for performing sum operations on a binary heap.
//! The heap calculator employs the concept of links, where each link contains a value, a stuck bit
//! for fault injection, and a mask for manipulating individual bits.
//!
//! The `HeapCalculator` struct represents a binary heap calculator, and the `Link` struct is used
//! for elements in the heap.
//!
//! ## Example
//!
//! ```rust
//! use rand::Rng;
//! use heap_calculator::{HeapCalculator, Stuck};
//!
//! // Create a new heap calculator with fault injection at a random position.
//! let mut heap_calculator: HeapCalculator<f64, u64> = HeapCalculator::new(4, Stuck::Zero);
//!
//! // Input values to the heap calculator.
//! let inputs = vec![1.0, 2.0, 3.0, 4.0];
//!
//! // Perform sum operation on all input values.
//! let result = heap_calculator.sum_all(&inputs);
//!
//! println!("Result: {}", result);
//! ```
//!
//! ## Stuck Types
//!
//! The `Stuck` enum defines different stuck-at fault types for injection.
//! - `Stuck::Zero`: Injects a fault where the bit is stuck at 0.
//! - `Stuck::One`: Injects a fault where the bit is stuck at 1.
//! - `Stuck::Transient`: Injects a transient fault (flips the bit).
//!
//! ## References
//!
//! This implementation is based on the binary heap sum operation and fault injection concept.
//!

use std::{
    fmt::Display,
    marker::PhantomData,
    ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, Not, Shr},
};

use rand::Rng;

use super::{Stuck, ToBits};

#[derive(Debug, Clone)]
pub struct HeapCalculator<T: Clone, U> {
    heap_vec: Vec<Link<T, U>>,
}

#[derive(Debug, Clone)]
pub struct Link<T: Clone, U> {
    value: T,
    stuck_bit: Option<Stuck>, // stuck 0,1, transient + index
    mask: Option<U>, // is a binary number with all 0s except a 1 in position idx (defined randomly later)

    marker: PhantomData<U>,
}

impl<T, U> Link<T, U>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U> + Default,
    U: BitOrAssign
        + BitAnd<Output = U>
        + BitOr<Output = U>
        + Not<Output = U>
        + BitAndAssign
        + Shr
        + Clone
        + Default
        + PartialEq,
{
    /// Creates a new `HeapCalculator` with fault injection at a random position.
    ///
    /// # Arguments
    ///
    /// * `dim` - The number of elements in the input.
    /// * `stuck` - The type of fault to inject.
    ///
    pub fn new(value: T, stuck_bit: Option<Stuck>) -> Self {
        //select the random bit to apply injection
        let idx = rand::thread_rng().gen_range(0..value.num_bits());
        // println!("idx: {}", idx);
        Link {
            stuck_bit,
            marker: PhantomData,
            mask: Some(value.create_mask(idx)),
            value,
        }
    }

    pub fn sum(&self, link: &Link<T, U>) -> T {
        if let Some(stuck) = &self.stuck_bit {
            let mut bits: U = self.value.get_bits(); //get the bit representation of the value

            match stuck {
                Stuck::Zero => bits &= !(self.mask.as_ref().unwrap().clone()),
                Stuck::One => bits |= self.mask.as_ref().unwrap().clone(),
                Stuck::Transient => bits = self.invert_bit_at(&bits, self.mask.clone()),
            };
            let new_val = T::from_bits(bits);
            return new_val + link.value.clone();
        }

        if let Some(stuck) = &link.stuck_bit {
            let mut bits: U = link.value.get_bits(); //get the bit representation of the value
            match stuck {
                Stuck::Zero => bits &= !(link.mask.as_ref().unwrap().clone()),
                Stuck::One => bits |= link.mask.as_ref().unwrap().clone(),
                Stuck::Transient => bits = self.invert_bit_at(&bits, link.mask.clone()),
            };
            let new_val = T::from_bits(bits);
            return new_val + self.value.clone();
        }

        self.value.clone() + link.value.clone()
    }

    /* pub fn product(&self, link: &Link<T, U>) -> T {
        if let Some(stuck) = &self.stuck_bit {
            let mut bits: U = self.value.get_bits(); //get the bit representation of the value

            match stuck {
                Stuck::Zero => bits &= !(self.mask.as_ref().unwrap().clone()),
                Stuck::One => bits |= self.mask.as_ref().unwrap().clone(),
                Stuck::Transient => bits = self.invert_bit_at(&bits, link),
            };

            let new_val = T::from_bits(bits);

            return new_val * link.value.clone();
        }

        if let Some(stuck) = &link.stuck_bit {
            let mut bits: U = link.value.get_bits(); //get the bit representation of the value

            match stuck {
                Stuck::Zero => bits &= !(self.mask.as_ref().unwrap().clone()),
                Stuck::One => bits |= self.mask.as_ref().unwrap().clone(),
                Stuck::Transient => bits = self.invert_bit_at(&bits, link),
            };
            let new_val = T::from_bits(bits);
            return new_val * self.value.clone();
        }

        self.value.clone() * link.value.clone()
    } */

    pub fn invert_bit_at(&self, bits: &U, mask: Option<U>) -> U {
        if (bits.clone() & mask.as_ref().unwrap().clone()) != U::default() {
            bits.clone() & !(mask.as_ref().unwrap().clone())
        } else {
            bits.clone() | mask.as_ref().unwrap().clone()
        }
    }
}

impl<T: Display, U: Display> HeapCalculator<T, U>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U> + Default,
    U: BitOrAssign
        + BitAnd<Output = U>
        + Not<Output = U>
        + BitAndAssign
        + Shr
        + Clone
        + Default
        + PartialEq
        + BitOr<Output = U>,
{
    pub fn new(dim: usize, stuck: Stuck) -> Self {
        //dim is the number of elements in the input
        let heap_length = 2 * dim;
        //create a new heap vector filled with Link struct
        let mut heap_vec: Vec<Link<T, U>> = vec![
            Link {
                value: T::default(),
                stuck_bit: None,
                mask: None,
                marker: PhantomData,
            };
            heap_length
        ];
        // println!("{}   {}", heap_vec.len(), dim);
        //choose the link to apply the injection
        let index = rand::thread_rng().gen_range(0..heap_length);
        heap_vec[index] = Link::new(T::default(), Some(stuck));

        // println!("stuck link: {}", index);

        HeapCalculator { heap_vec }
    }

    pub fn sum_all(&mut self, inputs: &[T]) -> T {
        let len = self.heap_vec.len();
        // println!("len: {}", len);
        //adding values to the heap
        for (i, val) in inputs.into_iter().enumerate() {
            self.heap_vec[i].value = val.clone();
        }
        //for the index
        let mut start: usize = 0;
        let number_of_levels: usize = log2(len);
        // println!("number_of_levels: {}", number_of_levels);

        for lv in 0..number_of_levels {
            // println!("lv: {}", lv);

            let lv_dim = len >> (lv + 1);
            // println!("lv_dim: {}", lv_dim);

            for i in (0..lv_dim).step_by(2) {
                // println!(
                //     "{} {}",
                //     self.heap_vec[start + i].value,
                //     self.heap_vec[start + i + 1].value
                // );
                let tmp = self.heap_vec[start + i].sum(&self.heap_vec[start + i + 1]);

                self.heap_vec[start + (len >> (lv + 1)) + i / 2].value = tmp;
            }
            start += lv_dim;
        }
        self.heap_vec[len - 2].value.clone()
    }

    /*  pub fn multiply_all(&mut self, inputs: &[T]) -> T {
        let len = self.heap_vec.len();
        println!("len: {}", len);
        for (i, val) in inputs.into_iter().enumerate() {
            self.heap_vec[i].value = val.clone();
        }
        let mut start: usize = 0;

        for lv in 0..log2(len) {
            println!("lv: {}", lv);
            for i in (0..len >> (lv + 1)).step_by(2) {
                println!(
                    "{} {}",
                    self.heap_vec[start + i].value,
                    self.heap_vec[start + i + 1].value
                );
                let tmp = self.heap_vec[start + i].product(&self.heap_vec[start + i + 1]);
                self.heap_vec[start + (len >> (lv + 1)) + i / 2].value = tmp;
            }
            start += (len >> (lv + 1));
        }
        self.heap_vec[len - 2].value.clone()
    } */
}
fn log2(len: usize) -> usize {
    let mut shifts = 0;
    let mut n = len;

    while n > 1 {
        n >>= 1; // shift on right
        shifts += 1;
    }

    shifts
}
