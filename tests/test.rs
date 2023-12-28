use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra::DMatrix;
use spiking_nn_resilience::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
use spiking_nn_resilience::snn::json_adapter::{InputJson, LayerWeightsJson, NeuronJson};
use spiking_nn_resilience::*;

#[test]
fn test_passthrough_nn() {
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
    nn.expect("Ciao").solve_multiple_vec_spike(spikes, 11);

    /*
      assert_eq!(
          nn.expect("Ciao").solve_multiple_vec_spike(spikes,11),
          vec![
              vec![1, 2, 3, 5, 6, 7],
              vec![2, 6, 7, 9],
              vec![2, 5, 6, 10, 11]
          ]
      );
    */
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

    println!(
        "{}",
        DMatrix::from_vec(3, 2, vec![0.0, 1.0, 0.5, 0.5, 0.0, 1.0,])
    );
    let spikes: Vec<(u128, Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7]),
        (1, vec![2, 6, 7, 9]),
        (2, vec![2, 5, 6, 10, 11]),
    ];
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    nn.expect("Error").solve_multiple_vec_spike(spikes, 11);

    let time2 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();

    println!(" executed in {} seconds", time2 - time);

    /*
      assert_eq!(
          nn.expect("Ciao").solve_multiple_vec_spike(spikes,11),
          vec![
              vec![1, 2, 3, 5, 6, 7],
              vec![2, 6, 7, 9],
              vec![2, 5, 6, 10, 11]
          ]
      );
    */
    return;
}

#[test]
fn test_single_thread() {
    let config = Configuration::new(2.0, 0.5, 2.1, 1.0);

    let nn = NN::<LeakyIntegrateFire>::new()
        .layer(
            vec![
                LifNeuron::from_conf(&config),
                LifNeuron::from_conf(&config),
                LifNeuron::from_conf(&config),
            ],
            DMatrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            DMatrix::from_vec(3, 3, vec![0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        )
        .unwrap()
        .layer(
            vec![LifNeuron::from_conf(&config), LifNeuron::from_conf(&config)],
            DMatrix::from_vec(3, 2, vec![0.0, 0.5, 0.0, 1.0, 0.5, 1.0]),
            DMatrix::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]),
        );

    let input = vec![1, 2, 5, 7, 8, 10, 11];
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let res = nn.expect("error in nn").solve_single_thread(input);

    let time2 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    println!(" executed in {} seconds", time2 - time);
    /*  assert_eq!(
        nn.expect("error ").solve_single_thread(input),
        vec![(2, vec![1.,1.]),
        (4, vec![1.,1.]),
        (7, vec![1.,1.])
        ]
    );
    */

    return;
}

#[test]
fn test_nn_multiple_layer() {
    let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0); // L0n0, L1n0, L2n0, L2n1
    let config_1 = Configuration::new(2.0, 0.5, 2.6, 1.0); // L0n1
    let config_2 = Configuration::new(1.7, 0.3, 3.4, 1.0); // L0n2
    let config_3 = Configuration::new(2.0, 0.8, 4.3, 1.0); //       L1n1

    let nn = NN::<LeakyIntegrateFire>::new()
        .layer(
            vec![
                LifNeuron::from_conf(&config_0),
                LifNeuron::from_conf(&config_1),
                LifNeuron::from_conf(&config_2),
            ],
            DMatrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            DMatrix::from_vec(3, 3, vec![0.0, -0.5, -1.0, 0.0, 0.0, -2.0, -0.5, 0.0, 0.0]),
        )
        .unwrap()
        .layer(
            vec![
                LifNeuron::from_conf(&config_0),
                LifNeuron::from_conf(&config_3),
            ],
            DMatrix::from_vec(3, 2, vec![2.0, 0.5, 3.0, 1.0, 0.0, 2.0]),
            DMatrix::from_vec(2, 2, vec![0.0, -1.0, 0.0, 0.0]),
        )
        .unwrap()
        .layer(
            vec![
                LifNeuron::from_conf(&config_0),
                LifNeuron::from_conf(&config_3),
            ],
            DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 0.5]),
            DMatrix::from_vec(2, 2, vec![0.0, 0.0, -1.0, 0.0]),
        )
        .unwrap()
        .layer(
            vec![
                LifNeuron::from_conf(&config_3), //4,3 3
                LifNeuron::from_conf(&config_0), //1,1 2
                LifNeuron::from_conf(&config_2), //3,4 1
                LifNeuron::from_conf(&config_1), //2,6 1
            ],
            DMatrix::from_vec(2, 4, vec![2.0, 1.0, 1.5, 0.5, 1.0, 0.0, 0.5, 0.5]),
            DMatrix::from_vec(
                4,
                4,
                vec![
                    0.0, -0.5, -1.0, 0.0, 0.0, 0.0, -0.5, 0.0, -0.5, 0.0, 0.0, -0.1, -0.1, -0.1,
                    -0.2, -0.1,
                ],
            ),
        );
    println!(
        "{}",
        DMatrix::from_vec(2, 4, vec![2.0, 1.0, 1.5, 0.5, 1.0, 0.0, 0.5, 0.5,]),
    );
    /*
    let spikes:Vec<(u128,Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7]),
        (1, vec![2, 6, 7, 8]),
        (2, vec![2, 5, 6]),
    ];

     */
    let spikes = vec![
        (0, vec![1, 2, 5, 7, 8, 10, 11]),
        (1, vec![1, 2, 5, 7, 8, 10, 11]),
        (2, vec![1, 2, 5, 7, 8, 10, 11]),
    ];
    nn.expect("Ciao").solve_multiple_vec_spike(spikes, 11);

    /*
      assert_eq!(
          nn.expect("Ciao").solve_multiple_vec_spike(spikes,11),
          vec![
              vec![1, 2, 3, 5, 6, 7],
              vec![2, 6, 7, 9],
              vec![2, 5, 6, 10, 11]
          ]
      );
    */
    return;
}

#[test]
fn test_nn_single_thread_complete() {
    let config_0 = Configuration::new(2.0, 0.5, 1.1, 1.0); // L0n0, L1n0, L2n0, L2n1
    let config_1 = Configuration::new(2.0, 0.5, 2.6, 1.0); // L0n1
    let config_2 = Configuration::new(1.7, 0.3, 3.4, 1.0); // L0n2
    let config_3 = Configuration::new(2.0, 0.8, 4.3, 1.0); //       L1n1

    let nn = NN::<LeakyIntegrateFire>::new()
        .layer(
            vec![
                LifNeuron::from_conf(&config_0),
                LifNeuron::from_conf(&config_1),
                LifNeuron::from_conf(&config_2),
            ],
            DMatrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            DMatrix::from_vec(3, 3, vec![0.0, -0.5, -1.0, 0.0, 0.0, -2.0, -0.5, 0.0, 0.0]),
        )
        .unwrap()
        .layer(
            vec![
                LifNeuron::from_conf(&config_0),
                LifNeuron::from_conf(&config_3),
            ],
            DMatrix::from_vec(3, 2, vec![2.0, 0.5, 3.0, 1.0, 0.0, 2.0]),
            DMatrix::from_vec(2, 2, vec![0.0, -1.0, 0.0, 0.0]),
        )
        .unwrap()
        .layer(
            vec![
                LifNeuron::from_conf(&config_0),
                LifNeuron::from_conf(&config_3),
            ],
            DMatrix::from_vec(2, 2, vec![1.0, 2.0, 2.0, 0.5]),
            DMatrix::from_vec(2, 2, vec![0.0, 0.0, -1.0, 0.0]),
        )
        .unwrap()
        .layer(
            vec![
                LifNeuron::from_conf(&config_3), //4,3 3
                LifNeuron::from_conf(&config_0), //1,1 2
                LifNeuron::from_conf(&config_2), //3,4 1
                LifNeuron::from_conf(&config_1), //2,6 1
            ],
            DMatrix::from_vec(2, 4, vec![2.0, 1.0, 1.5, 0.5, 1.0, 0.0, 0.5, 0.5]),
            DMatrix::from_vec(
                4,
                4,
                vec![
                    0.0, -0.5, -1.0, 0.0, 0.0, 0.0, -0.5, 0.0, -0.5, 0.0, 0.0, -0.1, -0.1, -0.1,
                    -0.2, -0.1,
                ],
            ),
        );
    let input = vec![1, 2, 5, 7, 8, 10, 11];
    nn.expect("Ciao").solve_single_thread(input);

    /*
      assert_eq!(
          nn.expect("Ciao").solve_multiple_vec_spike(spikes,11),
          vec![
              vec![1, 2, 3, 5, 6, 7],
              vec![2, 6, 7, 9],
              vec![2, 5, 6, 10, 11]
          ]
      );
    */
    return;
}

// Input from file
#[test]
fn test_nn_multiple_layer_from_file() {
    let mut nn =
        NeuronJson::read_from_file("./tests/layer_configuration.json", "./tests/weights.json");

    let input = InputJson::read_input_from_file("./tests/input_spikes.json");
    
    return;
}
