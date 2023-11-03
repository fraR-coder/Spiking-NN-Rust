
use nalgebra::DMatrix;
use spiking_nn_resilience::{*, lif::{Configuration, LeakyIntegrateFire, LifNeuron}, snn::model::{logic_circuit::{FullAdderTree, FullAdder}, LogicCircuit, heap::HeapAdder}};

#[test]
fn test_sum(){
    
    let inputs=vec![1.0,2.0,3.0,1.0,0.0,4.0];
    let mut heap_adder:HeapAdder<f64,u64 >=HeapAdder::new(2*(2 as u32).pow((( inputs.len() as f64).log2().ceil()) as u32) as usize, 1);
    println!("somma: {}", heap_adder.sum_all(&inputs))
}

#[test]

fn bit(){
    let num_10:f64 = 10.0;
    let num_20:f64 = 20.0;

    let bits_10 = num_10.to_bits();
    let bits_20 = num_20.to_bits();

    let xor_result = bits_10 ^ bits_20;

    // Trova il primo bit a 1 da destra a sinistra
    let changed_bit = (64 - xor_result.leading_zeros()) - 1;

    println!("Bit da cambiare: {}", changed_bit);
}