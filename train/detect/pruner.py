from torch_pruning import RTOSSPruner

pruner_config = {
    'pruning_algorithm': 'l1',
    'pruning_ratio': 0.5,
    'iterative_steps': 5,
    'sparsity_training_epochs': 10,
    'importance_score': 'magnitude',
}
