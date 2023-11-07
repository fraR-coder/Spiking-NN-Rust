use std::{
    fmt::Display,
    marker::PhantomData,
    ops::{Add, BitAndAssign, BitOrAssign, Mul, Not},
};

use rand::Rng;

use super::{logic_circuit::Stuck, ToBits};

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
    U: BitOrAssign + Not<Output = U> + BitAndAssign + Clone,
{
    pub fn new(value: T, stuck_bit: Option<Stuck>) -> Self {
        //select the random bit to apply injection
        let idx = rand::thread_rng().gen_range(0..value.num_bits());
        println!("idx: {}", idx);
        Link {
            stuck_bit,
            marker: PhantomData,
            mask: Some(value.create_mask(idx)),
            value,
        }
    }

    pub fn sum(&self, link: &Link<T, U>) -> T {
        if let Some(stuck) = &self.stuck_bit {
            let mut bits: U = self.value.get_bits(); //get the bit rapresentation of the value
            //chose the tpe of error
            match stuck {
                Stuck::Zero => bits &= !(self.mask.as_ref().unwrap().clone()),
                Stuck::One => bits |= self.mask.as_ref().unwrap().clone(),
                Stuck::Transient => {}
            };
            let new_val = T::from_bits(bits);
            return new_val + link.value.clone();
        }

        if let Some(stuck) = &link.stuck_bit {
            let mut bits: U = link.value.get_bits(); //get the bit rapresentation of the value

            match stuck {
                Stuck::Zero => bits &= !(link.mask.as_ref().unwrap().clone()),
                Stuck::One => bits |= link.mask.as_ref().unwrap().clone(),
                Stuck::Transient => {}
            };
            let new_val = T::from_bits(bits);
            return new_val + self.value.clone();
        }

        self.value.clone() + link.value.clone()
    }

    pub fn product(&self, link: &Link<T, U>) -> T {
        if let Some(stuck) = &self.stuck_bit {
            let mut bits: U = self.value.get_bits(); //get the bit rapresentation of the value

            match stuck {
                Stuck::Zero => bits &= !(self.mask.as_ref().unwrap().clone()),
                Stuck::One => bits |= self.mask.as_ref().unwrap().clone(),
                Stuck::Transient => {}
            };

            let new_val = T::from_bits(bits);

            return new_val * link.value.clone();
        }

        if let Some(stuck) = &link.stuck_bit {
            let mut bits: U = link.value.get_bits(); //get the bit rapresentation of the value

            match stuck {
                Stuck::Zero => bits &= !(self.mask.as_ref().unwrap().clone()),
                Stuck::One => bits |= self.mask.as_ref().unwrap().clone(),
                Stuck::Transient => {}
            };
            let new_val = T::from_bits(bits);
            return new_val * self.value.clone();
        }

        self.value.clone() * link.value.clone()
    }
}

impl<T: Display, U: Display> HeapCalculator<T, U>
where
    T: Add<Output = T> + Mul<Output = T> + Clone + ToBits<U> + Default,
    U: BitOrAssign + Not<Output = U> + BitAndAssign + Clone,
{
    pub fn new(dim: usize, stuck: Stuck) -> Self {
        //dim is the number of elemnts in the input
        let heap_length = 2 * dim +1 ; 
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
        println!("{}   {}", heap_vec.len(), dim);
        //choose the link to apply the injection
        let index = rand::thread_rng().gen_range(0..heap_length);
        heap_vec[index] = Link::new(T::default(), Some(stuck));

        println!("stuck link: {}", index);

        HeapCalculator { heap_vec }
    }

    /*
    pub fn sum_all(&mut self, inputs: &[T]) -> T {
    let len = self.heap_vec.len();

    // Inserisci gli elementi di input direttamente nell'heap se possibile.
    self.heap_vec.extend(inputs.iter().cloned());

    let mut start: usize = 0;
    let number_of_levels: usize = (len as f64).log2().ceil() as usize;

    for lv in 0..number_of_levels {
        let lv_dim = (len as f64 / (2.0 * 2_usize.pow(lv as u32) as f64)).ceil() as usize;

        for i in (0..lv_dim).step_by(2) {
            let left_idx = start + i;
            let right_idx = start + i + 1;
            let parent_idx = start + len / (2 * 2_usize.pow(lv as u32)) + i / 2;

            let left_val = self.heap_vec[left_idx].clone();
            let right_val = self.heap_vec[right_idx].clone();
            let tmp = left_val.sum(&right_val);

            self.heap_vec[parent_idx] = tmp;
        }
        start += lv_dim;
    }

    self.heap_vec[len - 2].clone()
}
 */
    pub fn sum_all(&mut self, inputs: &[T]) -> T {
        let len = self.heap_vec.len();
        println!("len: {}", len);
        //aggiungo i valori nell'heap
        for (i, val) in inputs.into_iter().enumerate() {
            self.heap_vec[i].value = val.clone();
        }
        //usata per muoversi nel vettore e riposizionare l'indice iniziale su cui voglio lavorare
        let mut start: usize = 0;
        let number_of_levels:usize=(len as f64).log2().ceil() as usize;

        //scorro i livelli dell'albero creato dall'heap
        for lv in 0..number_of_levels {
            println!("lv: {}", lv);

            //per ogni livello itera su un numero di elementi diversi,
            // al lv 0 sono tutti gli elementi di inputs, quindi la prima metà del vettore
            //all'aumentare del livello dimezzo il numero di dati su cui lavoro
            let lv_dim=((len as f64) / (2 * (2 as u32).pow(lv as u32)) as f64).ceil() as usize;

            //sommo le deu coppie di valori adiacenti quindi avanzo di due indici alla volta
            for i in (0..lv_dim).step_by(2)
            {
                println!(
                    "{} {}",
                    self.heap_vec[start + i].value,
                    self.heap_vec[start + i + 1].value
                );
                //somma la coppia di valori
                let tmp = self.heap_vec[start + i].sum(&self.heap_vec[start + i + 1]);

                self.heap_vec[start + len / (2 * (2 as u32).pow(lv as u32)) as usize + i / 2]
                    .value = tmp;
            }
            start += lv_dim ;
        }
        self.heap_vec[len - 2].value.clone()
    }


    pub fn multiply_all(&mut self, inputs: &[T]) -> T {
        let len = self.heap_vec.len();
        println!("len: {}", len);
        for (i, val) in inputs.into_iter().enumerate() {
            self.heap_vec[i].value = val.clone();
        }
        let mut start: usize = 0;

        for lv in (0..(len as f64).log2().ceil() as usize) {
            println!("lv: {}", lv);
            for i in (0..((len as f64) / (2 * (2 as u32).pow(lv as u32)) as f64).ceil() as usize)
                .step_by(2)
            {
                println!(
                    "{} {}",
                    self.heap_vec[start + i].value,
                    self.heap_vec[start + i + 1].value
                );
                let tmp = self.heap_vec[start + i].product(&self.heap_vec[start + i + 1]);
                self.heap_vec[start + len / (2 * (2 as u32).pow(lv as u32)) as usize + i / 2]
                    .value = tmp;
            }
            start += ((len as f64) / ((2.0 * (2 as u32).pow(lv as u32) as f64).ceil())) as usize;
        }
        self.heap_vec[len - 2].value.clone()
    }
}
