use std::{
    marker::PhantomData,
    ops::{Add, BitAndAssign, BitOrAssign, Mul, Not}, fmt::Display,
};

use rand::Rng;

use super::ToBits;

#[derive(Debug, Clone)]
pub struct HeapAdder<T: Clone , U> {
    heap_vec: Vec<AdderLink<T, U>>,
}

#[derive(Debug, Clone)]
pub struct AdderLink<T: Clone, U> {
    value: T,
    stuck_bit: Option<u8>, // stuck 0,1, transient + index
    mask: Option<U>,

    marker: PhantomData<U>,
}

impl<T, U> AdderLink<T, U>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U> + Default,
    U: BitOrAssign + Not<Output = U> + BitAndAssign + Clone,
{
    pub fn new(value: T, stuck_bit: Option<u8>) -> Self {
        let idx=rand::thread_rng().gen_range(0..value.num_bits());
        println!("idx: {}",idx);
        AdderLink {
            stuck_bit,
            marker: PhantomData,
            mask: Some(value.create_mask(11)),
            value,
        }
    }

    pub fn sum(&self, link: &AdderLink<T, U>) -> T {
        if let Some(bit_index) = self.stuck_bit {
            let mut bits: U = self.value.get_bits(); //get the bit rapresentation of the value

            if bit_index == 1 {
                // if stuck_at_bit_1
                bits |= self.mask.as_ref().unwrap().clone();
            } else if bit_index == 0 {
                bits &= !(self.mask.as_ref().unwrap().clone());
            } else {
                todo!();
            }

            let new_val = T::from_bits(bits);
            return new_val + link.value.clone();
        }

        if let Some(bit_index) = link.stuck_bit {
            let mut bits: U = link.value.get_bits(); //get the bit rapresentation of the value

            if bit_index == 1 {
                // if stuck_at_bit_1
                bits |= link.mask.as_ref().unwrap().clone();
            } else if bit_index == 0 {
                bits &= !(link.mask.as_ref().unwrap().clone());
            } else {
                todo!();
            }

            let new_val = T::from_bits(bits);
            return new_val + self.value.clone();
        }

        self.value.clone() + link.value.clone()
    }
}

impl<T: Display, U: Display> HeapAdder<T, U>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U> + Default,
    U: BitOrAssign + Not<Output = U> + BitAndAssign + Clone,
{
    pub fn new(dim: usize, stuck_bit: u8) -> Self {
        let mut heap_vec = Vec::with_capacity(dim);
        for i in 0..dim {
            heap_vec.push(AdderLink { value :T::default(), stuck_bit: None, mask: None, marker: PhantomData});
        }
        println!("{}   {}",heap_vec.len(),dim);
        let index = rand::thread_rng().gen_range(0..dim);
        heap_vec[index] =
            AdderLink::new(T::default(), Some(stuck_bit));

        println!("stuck link: {}", index );

        HeapAdder {heap_vec}
    }


    pub fn sum_all(&mut self, inputs: &[T])->T{
        let len= self.heap_vec.len();
        println!("len: {}", len);
        for (i,val) in inputs.into_iter().enumerate(){
                self.heap_vec[i].value = val.clone();
        }
        let mut start: usize = 0;

        for lv in (0.. (len as f64).log2().ceil() as usize) {
            println!("lv: {}", lv);
            for i in (0..((len as f64)/(2*(2 as u32).pow(lv as u32)) as f64).ceil() as usize).step_by(2){
                println!("{} {}", self.heap_vec[start+i].value, self.heap_vec[start+i+1].value);
                let tmp = self.heap_vec[start+i].sum(&self.heap_vec[start+i+1]);
                self.heap_vec[start + len/(2*(2 as u32).pow(lv as u32)) as usize + i/2].value = tmp ;
            }
            start += ((len as f64) / ((2.0 * (2 as u32).pow(lv as u32)as f64).ceil())) as usize;
        }
        self.heap_vec[len-2].value.clone()
    }
}
