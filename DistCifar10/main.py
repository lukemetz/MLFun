from theano import tensor as T
import theano
import numpy as np

from fuel.datasets.cifar10 import CIFAR10
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

from blocks.graph import ComputationGraph
from blocks.algorithms import GradientDescent
from blocks.model import Model

from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring

from blocks.extensions.plot import Plot

from cuboid.algorithms import NAG, AdaM
from blocks.main_loop import MainLoop

from cuboid.extensions import EpochProgress, LogToFile
from cuboid.extensions.distribute_extensions import DistributeUpdate, DistributeWhetlabFinish

from model import ModelHelper

from distribute import worker_from_url
from distribute import whetlab_make_next_jobs_func
import os

import whetlab

import logging
logging.basicConfig()

class Runner(object):
    def __init__(self, worker, experiment):
        config = experiment.get_by_result_id(int(worker.running_job))
        print config, worker.running_job

        # Data
        dataset = CIFAR10('train', flatten=False)
        test_dataset = CIFAR10('test', flatten=False)
        batch_size = 128

        scheme = ShuffledScheme(dataset.num_examples, batch_size)
        datastream = DataStream(dataset, iteration_scheme=scheme)

        test_scheme = ShuffledScheme(test_dataset.num_examples, batch_size)
        test_stream = DataStream(test_dataset, iteration_scheme=test_scheme)

        # Model
        m = ModelHelper(config)

        def score_func(mainloop):
            scores = mainloop.log.to_dataframe()["test_accur"].values
            return np.mean(np.sort(scores)[-4:-1])

        # Algorithm
        cg = ComputationGraph([m.cost])
        algorithm = GradientDescent(
                cost = m.cost, params=cg.parameters,
                step_rule = AdaM())

        job_name = os.path.basename(worker.running_job)
        update_path = (os.path.join(os.path.join(worker.path, "updates"), job_name))
        if not os.path.exists(update_path):
            os.mkdir(update_path)

        self.main_loop = MainLoop(
            algorithm,
            datastream,
            model = Model(m.cost),
            extensions=[
                Timing(),
                TrainingDataMonitoring(
                    [m.cost, m.accur], prefix="train", after_epoch=True)
                , DataStreamMonitoring(
                    [m.cost, m.accur],
                    test_stream,
                    prefix="test")
                , FinishAfter(after_n_epochs=1)
                , LogToFile(os.path.join(update_path, "log.csv"))
                , Printing()
                , EpochProgress(dataset.num_examples // batch_size + 1)
                , DistributeUpdate(worker, every_n_epochs=1)
                , DistributeWhetlabFinish(worker, experiment, score_func)
                #, Plot('cifar10',
                    #channels=[['train_cost', 'test_cost'], ['train_accur', 'test_accur']])
                ])
    def run(self):
        self.main_loop.run()

import sys
import pickle
def main():
    github_remote = "git@github.com:lukemetz/cifar10_sync.git"
    worker = worker_from_url(github_remote, path="cifar10_sync_"+sys.argv[1], name=sys.argv[1])
    print "getting whetlab experiment"
    experiment = whetlab.Experiment(name="Cifar10_2")

    while True:
        job = worker.get_next_job(whetlab_make_next_jobs_func(worker, experiment))
        print job
        if job == None:
            return
        r = Runner(worker, experiment)
        r.run()
if __name__ == "__main__":
    main()
