use nalgebra::DMatrix;
use spiking_nn_resilience::lif::{Configuration, LeakyIntegrateFire, LifNeuron};
use spiking_nn_resilience::*;
use spiking_nn_resilience::snn::Spike;

#[test]
fn test_passthrough_nn() {
    let config = Configuration::new(2.0, 0.5, 2.1, 1.0);

    let nn = NN::<LeakyIntegrateFire>::new().layer(
        vec![
            LifNeuron::from_conf(&config),
            LifNeuron::from_conf(&config),
            LifNeuron::from_conf(&config),
        ],
        DMatrix::from_vec(3, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]),
        DMatrix::from_vec(3, 3, vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]),
    );

    let spikes:Vec<(u128,Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7]),
        (1, vec![2, 6, 7, 9]),
        (2, vec![2, 5, 6, 10, 11]),
    ];
    nn.expect("Ciao").solve_multiple_vec_spike(spikes,11);

    /*
        assert_eq!(
            nn.expect("Ciao").solve_multiple_vec_spike(spikes,11),
            vec![
                vec![1, 2, 3, 5, 6, 7],
                vec![2, 6, 7, 9],
                vec![2, 5, 6, 10, 11]
            ]
        );
      */  return;
}

#[test]
fn test_nn_single_layer() {
    let config = Configuration::new(2.0, 0.5, 2.1, 1.0);

    let nn = NN::<LeakyIntegrateFire>::new().layer(
        vec![
            LifNeuron::from_conf(&config),
            LifNeuron::from_conf(&config),
            LifNeuron::from_conf(&config),
        ],
        DMatrix::from_vec(3, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]),
        DMatrix::from_vec(3, 3, vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0]),
    ).unwrap().layer(
        vec![
            LifNeuron::from_conf(&config),
            LifNeuron::from_conf(&config),
        ],
        DMatrix::from_vec(3, 2, vec![
            0.0, 1.0,
            0.5, 0.5,
            0.0, 1.0,
        ]),
        DMatrix::from_vec(2, 2, vec![
            0.0, 0.0,
            0.0, 0.0,
        ]),
    );

    println!("{}",DMatrix::from_vec(3, 2, vec![
        0.0, 1.0,
        0.5, 0.5,
        0.0, 1.0,
    ]));
    let spikes:Vec<(u128,Vec<u128>)> = vec![
        (0, vec![1, 2, 3, 5, 6, 7]),
        (1, vec![2, 6, 7, 9]),
        (2, vec![2, 5, 6, 10, 11]),
    ];
    nn.expect("Error").solve_multiple_vec_spike(spikes,11);

    /*
        assert_eq!(
            nn.expect("Ciao").solve_multiple_vec_spike(spikes,11),
            vec![
                vec![1, 2, 3, 5, 6, 7],
                vec![2, 6, 7, 9],
                vec![2, 5, 6, 10, 11]
            ]
        );
      */  return;
}