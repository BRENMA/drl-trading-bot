exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'retry_type': 'reset',
            'auto_reset': True,
            'step_timeout': None,
            'reset_timeout': None,
            'retry_waiting_time': 0.1,
            'cfg_type': 'BaseEnvManagerDict',
            'type': 'base'
        },
        'stop_value': 2,
        'type': 'stocks-v0',
        'import_names': ['dizoo.gym_anytrading.envs.stocks_env'],
        'collector_env_num': 8,
        'evaluator_env_num': 8,
        'env_id': 'stocks-v0',
        'n_evaluator_episode': 8,
        'eps_length': 253,
        'window_size': 20,
        'save_path': './fig/',
        'stocks_data_filename': 'STONKS',
        'train_range': None,
        'test_range': None
    },
    'policy': {
        'model': {
            'obs_shape': 62,
            'action_shape': 5,
            'encoder_hidden_size_list': [128],
            'head_layer_num': 1,
            'dueling': True
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'multi_gpu': False,
            'update_per_collect': 10,
            'batch_size': 64,
            'learning_rate': 0.001,
            'target_update_freq': 100,
            'ignore_done': True
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            },
            'unroll_len': 1,
            'n_sample': 64
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1000,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'type': 'trading_interaction',
                'import_names': ['dizoo.gym_anytrading.worker'],
                'n_episode': 8,
                'stop_value': 2
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 100000,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict'
            },
            'eps': {
                'type': 'exp',
                'start': 0.95,
                'end': 0.1,
                'decay': 50000
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'type': 'dqn',
        'cuda': True,
        'on_policy': False,
        'priority': False,
        'priority_IS_weight': False,
        'discount_factor': 0.99,
        'nstep': 5,
        'cfg_type': 'DQNPolicyDict'
    },
    'exp_name': 'stocks_dqn_seed0_221227_230358',
    'seed': 0
}
