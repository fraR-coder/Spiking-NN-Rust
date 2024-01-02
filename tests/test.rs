// use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::DMatrix;
use spiking_nn_resilience::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
use spiking_nn_resilience::snn::json_adapter::{InputJson, NeuronJson, ResilienceJson};
use spiking_nn_resilience::*;
use spiking_nn_resilience::snn::model::Stuck;
use spiking_nn_resilience::snn::resilience::Resilience;

#[test]
fn test_pass_through_nn() {
    let config = Configuration::new(2.0, 0.5, 2.1, 1.0);

    let nn = NN::<LeakyIntegrateFire>::new().layer(
        vec![
            LifNeuron::from_conf(&config),
            LifNeuron::from_conf(&config),
            LifNeuron::from_conf(&config),
        ],
        DMatrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        DMatrix::from_vec(3, 3, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    );

    let spikes: Vec<(u128, Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7]),
        (1, vec![2, 6, 7, 9]),
        (2, vec![2, 5, 6, 10, 11]),
    ];
    let res = nn.expect("Error").solve_multiple_vec_spike(spikes).lock().unwrap().clone();
      assert_eq!(
          res,
          vec![
              (0, vec![2, 3, 4, 6, 7, 8]),
              (1, vec![3, 7, 8, 10]),
              (2, vec![3, 6, 7, 11, 12])
          ]
      );
    return;
}

#[test]
fn test_nn_single_layer() {
    let config = Configuration::new(2.0, 0.5, 2.1, 1.0);

    let nn = NN::<LeakyIntegrateFire>::new()
        .layer(
            vec![
                LifNeuron::from_conf(&config),
                LifNeuron::from_conf(&config),
                LifNeuron::from_conf(&config),
            ],
            DMatrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            DMatrix::from_vec(3, 3, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .unwrap()
        .layer(
            vec![LifNeuron::from_conf(&config), LifNeuron::from_conf(&config)],
            DMatrix::from_vec(3, 2, vec![0.0, 1.0, 0.5, 0.5, 0.0, 1.0]),
            DMatrix::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]),
        );

    let spikes: Vec<(u128, Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7]),
        (1, vec![2, 6, 7, 9]),
        (2, vec![2, 5, 6, 10, 11]),
    ];

    let res = nn.expect("Error").solve_multiple_vec_spike(spikes).lock().unwrap().clone();
    assert_eq!(
        res,
        vec![
            (0, vec![4, 7, 8, 9, 11, 13]),
            (1, vec![3, 4, 7, 8, 12, 13]),
        ]
    );
    return;
}


#[test]
fn test_nn_multiple_layer() {
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

    let spikes:Vec<(u128,Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7, 10, 13, 14, 15]),
        (1, vec![2, 6, 7, 8, 10, 11, 12]),
        (2, vec![2, 5, 6, 7, 9, 10, 15]),
    ];

    let res = nn.expect("Error").solve_multiple_vec_spike(spikes).lock().unwrap().clone();
    assert_eq!(
        res,
        vec![
            (0, vec![6, 10, 12, 13, 15, 16, 18, 19]),
            (1, vec![5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            (2, vec![]),
            (3, vec![6, 10, 13, 16, 19])
        ]
    );
    return;
}

#[test]
fn test_nn_multiple_layer_resilience() {
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

    let spikes:Vec<(u128,Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7, 10, 13, 14, 15]),
        (1, vec![2, 6, 7, 8, 10, 11, 12]),
        (2, vec![2, 5, 6, 7, 9, 10, 15]),
    ];

    let configuration: Resilience = Resilience::new(vec!["Full adder".to_string()], Stuck::One, 1);

    let res = configuration.execute_resilience_test(nn.clone().unwrap(),spikes).lock().unwrap().clone();
    assert_eq!(
        res,
        vec![
            (0, vec![6, 10, 12, 13, 15, 16, 18, 19]),
            (1, vec![5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            (2, vec![]),
            (3, vec![6, 10, 13, 16, 19])
        ]
    );
    return;
}


// Input from file
#[test]
fn test_nn_multiple_layer_from_file() {
    let nn =
        NeuronJson::read_from_file("./tests/layers.json", "./tests/weights.json", "./tests/configurations.json");

    let input = InputJson::read_input_from_file("./tests/input_spikes.json");

    // let configuration: Resilience = Resilience::new(vec!["Neurons".to_string()], Stuck::One, 1000);
    let configuration: Result<Resilience,String> = ResilienceJson::read_from_file("./tests/resilience.json").expect("Errore lettura file").to_resilience();
    let res = configuration.ok().unwrap().execute_resilience_test(nn.clone().unwrap(),input).lock().unwrap().clone();
    assert_eq!(
        res,
        vec![
            (0, vec![6, 10, 12, 13, 15, 16, 18, 19]),
            (1, vec![5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
            (2, vec![]),
            (3, vec![6, 10, 13, 16, 19])
        ]
    );
    return;
}
#[test]
fn test_nn_multiple_layer_from_file2() {
    let nn =
        NeuronJson::read_from_file("src/configuration/layers2.json", "src/configuration/weights2.json", "src/configuration/configurations2.json");

    let input = InputJson::read_input_from_file("src/configuration/input_spikes2.json");

    // let configuration: Resilience = Resilience::new(vec!["Neurons".to_string()], Stuck::One, 1000);
    let configuration: Result<Resilience,String> = ResilienceJson::read_from_file("src/configuration/resilience2.json").expect("Errore lettura file").to_resilience();
    let res = configuration.ok().unwrap().execute_resilience_test(nn.clone().unwrap(),input).lock().unwrap().clone();
    assert_eq!(
        res,
        vec![
            (0, vec![12, 17, 21, 25, 29]),
            (1, vec![12, 17, 21, 25, 29]),
            (2, vec![12, 17, 21, 25, 29]),
            (3, vec![12, 17, 21, 23, 25, 27, 29]),
            (4, vec![12, 17, 21, 23, 25, 27, 29]),
            (5, vec![12, 17, 21, 25, 29]),
            (6, vec![]),
            (7, vec![25, 29])
        ]
    );    return;
}

#[test]
fn test_all_combinations_() {
    let nn =
        NeuronJson::read_from_file("src/configuration/layers2.json", "src/configuration/weights2.json", "src/configuration/configurations2.json");

    let input = InputJson::read_input_from_file("src/configuration/input_spikes2.json");

    // let configuration: Resilience = Resilience::new(vec!["Neurons".to_string()], Stuck::One, 1000);
    let components = vec!["full adder".to_string(), "neuron".to_string(), "comparator".to_string(), "vmem".to_string()];
    let stucks = vec![Stuck::Zero, Stuck::One, Stuck::Transient];
    let mut res= vec![];
    for component in components{
        for stuck in stucks.clone(){
            let configuration: Resilience = Resilience::new(vec![component.clone()], stuck, 10000);

            // let configuration: Result<Resilience,String> = ResilienceJson::read_from_file("src/configuration/resilience2.json").expect("Errore lettura file").to_resilience();
            res = configuration.execute_resilience_test(nn.clone().unwrap(),input.clone()).lock().unwrap().clone();
        }
    }

    assert_eq!(
        res,
        vec![
            (0, vec![12, 17, 21, 25, 29]),
            (1, vec![12, 17, 21, 25, 29]),
            (2, vec![12, 17, 21, 25, 29]),
            (3, vec![12, 17, 21, 23, 25, 27, 29]),
            (4, vec![12, 17, 21, 23, 25, 27, 29]),
            (5, vec![12, 17, 21, 25, 29]),
            (6, vec![]),
            (7, vec![25, 29])
        ]
    );    return;
}
// #[test]
// fn test_fun_execute_resilience() {
//     let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0); // L0n0, L1n0, L2n0, L2n1
//     let config_1 = Configuration::new(2.0, 0.5, 2.6, 1.0); // L0n1
//     let config_2 = Configuration::new(1.7, 0.3, 3.4, 1.0); // L0n2
//     let config_3 = Configuration::new(2.0, 0.8, 4.3, 1.0); //       L1n1
//
//     let nn = NN::<LeakyIntegrateFire>::new().layer(
//         vec![
//             LifNeuron::from_conf(&config_0),
//             LifNeuron::from_conf(&config_1),
//             LifNeuron::from_conf(&config_2),
//         ],
//         DMatrix::from_vec(3, 3, vec![
//             1.0, 0.0, 0.0,
//             0.0, 1.0, 0.0,
//             0.0, 0.0, 1.0]),
//         DMatrix::from_vec(3, 3, vec![
//             0.0, -0.5, -1.0,
//             0.0, 0.0, -2.0,
//             -0.5, 0.0, 0.0]),
//     ).unwrap().layer(
//         vec![
//             LifNeuron::from_conf(&config_0),
//             LifNeuron::from_conf(&config_3),
//         ],
//         DMatrix::from_vec(3, 2, vec![
//             2.0, 0.5,
//             3.0, 1.0,
//             0.0, 2.0]),
//         DMatrix::from_vec(2, 2, vec![
//             0.0, -1.0,
//             0.0, 0.0]),
//     ).unwrap().layer(
//         vec![
//             LifNeuron::from_conf(&config_0),
//             LifNeuron::from_conf(&config_3),
//         ],
//         DMatrix::from_vec(2, 2, vec![
//             1.0, 2.0,
//             2.0, 0.5]),
//         DMatrix::from_vec(2, 2, vec![
//             0.0, 0.0,
//             -1.0, 0.0]),
//     ).unwrap().layer(
//         vec![
//             LifNeuron::from_conf(&config_3), //4,3 3
//             LifNeuron::from_conf(&config_0), //1,1 2
//             LifNeuron::from_conf(&config_2), //3,4 1
//             LifNeuron::from_conf(&config_1), //2,6 1
//         ],
//         DMatrix::from_vec(2,4,vec![
//             2.0,1.0,1.5,0.5,
//             1.0,0.0,0.5,0.5,
//         ]),
//         DMatrix::from_vec(4,4,vec![
//             0.0, -0.5, -1.0, 0.0,
//             0.0, 0.0, -0.5, 0.0,
//             -0.5, 0.0, 0.0, -0.1,
//             -0.1, -0.1, -0.2, -0.1,
//         ])
//     );
//     /*
//     let spikes:Vec<(u128,Vec<u128>)> = vec![
//         (0, vec![1, 2, 3, 5, 6, 7]),
//         (1, vec![2, 6, 7, 8]),
//         (2, vec![2, 5, 6]),
//     ];
//
//      */
//     let spikes=vec![
//         (0, vec![1,2,5,7,8,10,11]),
//         (1, vec![1,2,5,7,8,10,11]),
//         (2, vec![1,2,5,7,8,10,11]),
//     ];
//     let configuration: Resilience = Resilience::new(vec!["Full adder".to_string()], Stuck::One, 1);
//
//     configuration.execute_resilience_test(nn.clone().unwrap(),spikes);
// }


// #[test]
// fn test_resilience_for_logic_ciruits() {
//     let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0); // L0n0, L1n0, L2n0, L2n1
//     let config_1 = Configuration::new(2.0, 0.5, 2.6, 1.0); // L0n1
//     let config_2 = Configuration::new(1.7, 0.3, 3.4, 1.0); // L0n2
//     let config_3 = Configuration::new(2.0, 0.8, 4.3, 1.0); //       L1n1
//
//     let nn = NN::<LeakyIntegrateFire>::new().layer(
//         vec![
//             LifNeuron::from_conf(&config_0),
//             LifNeuron::from_conf(&config_1),
//             LifNeuron::from_conf(&config_2),
//         ],
//         DMatrix::from_vec(3, 3, vec![
//             1.0, 0.0, 0.0,
//             0.0, 1.0, 0.0,
//             0.0, 0.0, 1.0]),
//         DMatrix::from_vec(3, 3, vec![
//             0.0, -0.5, -1.0,
//             0.0, 0.0, -2.0,
//             -0.5, 0.0, 0.0]),
//     ).unwrap().layer(
//         vec![
//             LifNeuron::from_conf(&config_0),
//             LifNeuron::from_conf(&config_3),
//         ],
//         DMatrix::from_vec(3, 2, vec![
//             2.0, 0.5,
//             3.0, 1.0,
//             0.0, 2.0]),
//         DMatrix::from_vec(2, 2, vec![
//             0.0, -1.0,
//             0.0, 0.0]),
//     ).unwrap().layer(
//         vec![
//             LifNeuron::from_conf(&config_0),
//             LifNeuron::from_conf(&config_3),
//         ],
//         DMatrix::from_vec(2, 2, vec![
//             1.0, 2.0,
//             2.0, 0.5]),
//         DMatrix::from_vec(2, 2, vec![
//             0.0, 0.0,
//             -1.0, 0.0]),
//     ).unwrap().layer(
//         vec![
//             LifNeuron::from_conf(&config_3), //4,3 3
//             LifNeuron::from_conf(&config_0), //1,1 2
//             LifNeuron::from_conf(&config_2), //3,4 1
//             LifNeuron::from_conf(&config_1), //2,6 1
//         ],
//         DMatrix::from_vec(2,4,vec![
//             2.0,1.0,1.5,0.5,
//             1.0,0.0,0.5,0.5,
//         ]),
//         DMatrix::from_vec(4,4,vec![
//             0.0, -0.5, -1.0, 0.0,
//             0.0, 0.0, -0.5, 0.0,
//             -0.5, 0.0, 0.0, -0.1,
//             -0.1, -0.1, -0.2, -0.1,
//         ])
//     );
//     /*
//     let spikes:Vec<(u128,Vec<u128>)> = vec![
//         (0, vec![1, 2, 3, 5, 6, 7]),
//         (1, vec![2, 6, 7, 8]),
//         (2, vec![2, 5, 6]),
//     ];
//
//      */
//     let spikes=vec![
//         (0, vec![1,2,5,7,8,10,11]),
//         (1, vec![1,2,5,7,8,10,11]),
//         (2, vec![1,2,5,7,8,10,11]),
//     ];
//     let configuration: Resilience = Resilience::new(vec!["Full adder".to_string()], Stuck::Transient, 100);
//     // let configuration: Resilience = Resilience::new(vec!["full adder".to_string(), "Neurons".to_string()], Stuck::One, 1000);
//     // let configuration: Resilience = Resilience::new(vec!["full adder".to_string(), "Neurons".to_string()], Stuck::One, 1000);
//
//     configuration.execute_resilience_test(nn.clone().unwrap(),spikes);
// }

// #[test]
// fn test_single_thread() {
//     let config = Configuration::new(2.0, 0.5, 2.1, 1.0);
//
//     let nn = NN::<LeakyIntegrateFire>::new()
//         .layer(
//             vec![
//                 LifNeuron::from_conf(&config),
//                 LifNeuron::from_conf(&config),
//                 LifNeuron::from_conf(&config),
//             ],
//             DMatrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
//             DMatrix::from_vec(3, 3, vec![0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
//         )
//         .unwrap()
//         .layer(
//             vec![LifNeuron::from_conf(&config), LifNeuron::from_conf(&config)],
//             DMatrix::from_vec(3, 2, vec![0.0, 0.5, 0.0, 1.0, 0.5, 1.0]),
//             DMatrix::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]),
//         );
//
//     let input = vec![1, 2, 5, 7, 8, 10, 11];
//     let time = SystemTime::now()
//         .duration_since(UNIX_EPOCH)
//         .unwrap()
//         .as_nanos();
//     let res = nn.expect("error in nn").solve_single_thread(input);
//     // println!("{:?}",res);
//     let time2 = SystemTime::now()
//         .duration_since(UNIX_EPOCH)
//         .unwrap()
//         .as_nanos();
//
//     println!(" executed in {} seconds", time2 - time);
//     /*  assert_eq!(
//         nn.expect("error ").solve_single_thread(input),
//         vec![(2, vec![1.,1.]),
//         (4, vec![1.,1.]),
//         (7, vec![1.,1.])
//         ]
//     );
//     */
//
//     return;
// }

// #[test]
// fn test_nn_single_thread_complete() {
//     let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0); // L0n0, L1n0, L2n0, L2n1
//     let config_1 = Configuration::new(2.0, 0.5, 2.6, 1.0); // L0n1
//     let config_2 = Configuration::new(1.7, 0.3, 3.4, 1.0); // L0n2
//     let config_3 = Configuration::new(2.0, 0.8, 4.3, 1.0); //       L1n1
//
//     let nn = NN::<LeakyIntegrateFire>::new().layer(
//         vec![
//             LifNeuron::from_conf(&config_0),
//             LifNeuron::from_conf(&config_1),
//             LifNeuron::from_conf(&config_2),
//         ],
//         DMatrix::from_vec(3, 3, vec![
//             1.0, 0.0, 0.0,
//             0.0, 1.0, 0.0,
//             0.0, 0.0, 1.0]),
//         DMatrix::from_vec(3, 3, vec![
//             0.0, -0.5, -1.0,
//             0.0, 0.0, -2.0,
//             -0.5, 0.0, 0.0]),
//     ).unwrap().layer(
//         vec![
//             LifNeuron::from_conf(&config_0),
//             LifNeuron::from_conf(&config_3),
//         ],
//         DMatrix::from_vec(3, 2, vec![
//             2.0, 0.5,
//             3.0, 1.0,
//             0.0, 2.0]),
//         DMatrix::from_vec(2, 2, vec![
//             0.0, -1.0,
//             0.0, 0.0]),
//     ).unwrap().layer(
//         vec![
//             LifNeuron::from_conf(&config_0),
//             LifNeuron::from_conf(&config_3),
//         ],
//         DMatrix::from_vec(2, 2, vec![
//             1.0, 2.0,
//             2.0, 0.5]),
//         DMatrix::from_vec(2, 2, vec![
//             0.0, 0.0,
//             -1.0, 0.0]),
//     ).unwrap().layer(
//         vec![
//             LifNeuron::from_conf(&config_3), //4,3 3
//             LifNeuron::from_conf(&config_0), //1,1 2
//             LifNeuron::from_conf(&config_2), //3,4 1
//             LifNeuron::from_conf(&config_1), //2,6 1
//         ],
//         DMatrix::from_vec(2,4,vec![
//             2.0,1.0,1.5,0.5,
//             1.0,0.0,0.5,0.5,
//         ]),
//         DMatrix::from_vec(4,4,vec![
//             0.0, -0.5, -1.0, 0.0,
//             0.0, 0.0, -0.5, 0.0,
//             -0.5, 0.0, 0.0, -0.1,
//             -0.1, -0.1, -0.2, -0.1,
//         ])
//     );
//     let input = vec![1, 2, 5, 7, 8, 10, 11];
//     nn.expect("Ciao").solve_single_thread(input);
//
//     /*
//       assert_eq!(
//           nn.expect("Ciao").solve_multiple_vec_spike(spikes,11),
//           vec![
//               vec![1, 2, 3, 5, 6, 7],
//               vec![2, 6, 7, 9],
//               vec![2, 5, 6, 10, 11]
//           ]
//       );
//     */
//     return;
// }
