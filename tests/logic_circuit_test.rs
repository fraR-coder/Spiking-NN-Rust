use std::rc::Rc;

use nalgebra::DMatrix;

use spiking_nn_resilience::{*, lif::{Configuration, LeakyIntegrateFire, LifNeuron}, snn::model::{logic_circuit::{FullAdderTree, FullAdder, Stuck}, LogicCircuit}};


fn create_nn()->NN<LeakyIntegrateFire>{
    let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0); // L0n0, L1n0, L2n0, L2n1
    let config_1 = Configuration::new(2.0, 0.5, 2.6, 1.0); // L0n1
    let config_2 = Configuration::new(1.7, 0.3, 3.4, 1.0); // L0n2
    let config_3 = Configuration::new(2.0, 0.8, 4.3, 1.0); //       L1n1

    let nn = NN::<LeakyIntegrateFire>::new().layer(
        vec![
            LifNeuron::from_conf(&config_0),
            LifNeuron::from_conf(&config_1),
            LifNeuron::from_conf(&config_2),
        ],
        DMatrix::from_vec(3, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]),
        DMatrix::from_vec(3, 3, vec![
            0.0, -0.5, -1.0,
            0.0, 0.0, -2.0,
            -0.5, 0.0, 0.0]),
    ).unwrap().layer(
        vec![
            LifNeuron::from_conf(&config_0),
            LifNeuron::from_conf(&config_3),
        ],
        DMatrix::from_vec(3, 2, vec![
            2.0, 0.5,
            3.0, 1.0,
            0.0, 2.0]),
        DMatrix::from_vec(2, 2, vec![
            0.0, -1.0,
            0.0, 0.0]),
    ).unwrap().layer(
        vec![
            LifNeuron::from_conf(&config_0),
            LifNeuron::from_conf(&config_3),
        ],
        DMatrix::from_vec(2, 2, vec![
            1.0, 2.0,
            2.0, 0.5]),
        DMatrix::from_vec(2, 2, vec![
            0.0, 0.0,
            -1.0, 0.0]),
    ).unwrap().layer(
        vec![
            LifNeuron::from_conf(&config_3), //4,3 3
            LifNeuron::from_conf(&config_0), //1,1 2
            LifNeuron::from_conf(&config_2), //3,4 1
            LifNeuron::from_conf(&config_1), //2,6 1
        ],
        DMatrix::from_vec(2,4,vec![
            2.0,1.0,1.5,0.5,
            1.0,0.0,0.5,0.5,
        ]),
        DMatrix::from_vec(4,4,vec![
            0.0, -0.5, -1.0, 0.0,
            0.0, 0.0, -0.5, 0.0,
            -0.5, 0.0, 0.0, -0.1,
            -0.1, -0.1, -0.2, -0.1,
        ])
    );
    println!("{}",DMatrix::from_vec(2,4,vec![
        2.0,1.0,1.5,0.5,
        1.0,0.0,0.5,0.5,
    ]),);
    /*
    let spikes:Vec<(u128,Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7]),
        (1, vec![2, 6, 7, 8]),
        (2, vec![2, 5, 6]),
    ];

     */
    let spikes=vec![
        (0, vec![1,2,5,7,8,10,11]),
        (1, vec![1,2,5,7,8,10,11]),
        (2, vec![1,2,5,7,8,10,11]),
    ];

    return nn.unwrap();
}

#[test]
fn test_full_adder(){

    //let mut nn=create_nn();
    let inputs= Rc::new(vec![3.0,1.0,2.0,4.0,5.0]);


   

    let mut full_adder_tree:FullAdderTree<f64,u64>= FullAdderTree::new(inputs.len());
    
    

    
    let result=full_adder_tree.solve_tree_addition(Rc::clone(&inputs), Stuck::One);


    //full_adder_tree.get_full_adder_mut(0).unwrap().set_error_selector(Some((1,30))); //error in i1, bit 1

    

    println!("result: {} ", result);



    


    
}use std::{rc::Rc, result};

use nalgebra::DMatrix;
use spiking_nn_resilience::{*, lif::{Configuration, LeakyIntegrateFire, LifNeuron}, snn::model::{logic_circuit::{FullAdderTree, FullAdder}, LogicCircuit}};


fn create_nn()->NN<LeakyIntegrateFire>{
    let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0); // L0n0, L1n0, L2n0, L2n1
    let config_1 = Configuration::new(2.0, 0.5, 2.6, 1.0); // L0n1
    let config_2 = Configuration::new(1.7, 0.3, 3.4, 1.0); // L0n2
    let config_3 = Configuration::new(2.0, 0.8, 4.3, 1.0); //       L1n1

    let nn = NN::<LeakyIntegrateFire>::new().layer(
        vec![
            LifNeuron::from_conf(&config_0),
            LifNeuron::from_conf(&config_1),
            LifNeuron::from_conf(&config_2),
        ],
        DMatrix::from_vec(3, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]),
        DMatrix::from_vec(3, 3, vec![
            0.0, -0.5, -1.0,
            0.0, 0.0, -2.0,
            -0.5, 0.0, 0.0]),
    ).unwrap().layer(
        vec![
            LifNeuron::from_conf(&config_0),
            LifNeuron::from_conf(&config_3),
        ],
        DMatrix::from_vec(3, 2, vec![
            2.0, 0.5,
            3.0, 1.0,
            0.0, 2.0]),
        DMatrix::from_vec(2, 2, vec![
            0.0, -1.0,
            0.0, 0.0]),
    ).unwrap().layer(
        vec![
            LifNeuron::from_conf(&config_0),
            LifNeuron::from_conf(&config_3),
        ],
        DMatrix::from_vec(2, 2, vec![
            1.0, 2.0,
            2.0, 0.5]),
        DMatrix::from_vec(2, 2, vec![
            0.0, 0.0,
            -1.0, 0.0]),
    ).unwrap().layer(
        vec![
            LifNeuron::from_conf(&config_3), //4,3 3
            LifNeuron::from_conf(&config_0), //1,1 2
            LifNeuron::from_conf(&config_2), //3,4 1
            LifNeuron::from_conf(&config_1), //2,6 1
        ],
        DMatrix::from_vec(2,4,vec![
            2.0,1.0,1.5,0.5,
            1.0,0.0,0.5,0.5,
        ]),
        DMatrix::from_vec(4,4,vec![
            0.0, -0.5, -1.0, 0.0,
            0.0, 0.0, -0.5, 0.0,
            -0.5, 0.0, 0.0, -0.1,
            -0.1, -0.1, -0.2, -0.1,
        ])
    );
    println!("{}",DMatrix::from_vec(2,4,vec![
        2.0,1.0,1.5,0.5,
        1.0,0.0,0.5,0.5,
    ]),);
    /*
    let spikes:Vec<(u128,Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7]),
        (1, vec![2, 6, 7, 8]),
        (2, vec![2, 5, 6]),
    ];

     */
    let spikes=vec![
        (0, vec![1,2,5,7,8,10,11]),
        (1, vec![1,2,5,7,8,10,11]),
        (2, vec![1,2,5,7,8,10,11]),
    ];

    return nn.unwrap();
}

#[test]
fn test_full_adder(){

    let mut nn=create_nn();
    let inputs= Rc::new(vec![3.0,1.0]);


    let mut full_adder_tree:FullAdderTree<f64,u64>= FullAdderTree::new(inputs.len());

    println!("adders: {}", full_adder_tree.get_num_full_adders());

    
    let result=full_adder_tree.solve_tree_addition(Rc::clone(&inputs), false);


    full_adder_tree.get_full_adder_mut(0).unwrap().set_error_selector(Some((1,30))); //error in i1, bit 1

    

    let res2=full_adder_tree.solve_tree_addition(Rc::clone(&inputs), false);

    let res3=full_adder_tree.solve_tree_addition(Rc::clone(&inputs), true);

    println!("result: {}   with stuck0:{},  with stuck1:{}", result,res2,res3);



    


    
}