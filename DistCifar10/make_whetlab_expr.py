import whetlab

parameters = {
        'n_l0': {'min':10, 'max':100, 'type': 'integer'},
        'n_l1': {'min':10, 'max':100, 'type': 'integer'},
        'n_l2': {'min':10, 'max':100, 'type': 'integer'},
        'n_l3': {'min':10, 'max':100, 'type': 'integer'},
        'n_l4': {'min':10, 'max':100, 'type': 'integer'},
        'n_l5': {'min':10, 'max':100, 'type': 'integer'}
        }

outcome = {'name': 'accuracy'}
expr = whetlab.Experiment(name="Cifar10_2",
        description="frompython",
        parameters=parameters,
        outcome=outcome)

