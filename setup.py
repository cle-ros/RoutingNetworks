from setuptools import setup

setup(
    name='PytorchRouting',
    version='0.3',
    python_requires='>=3.5',
    packages=['PytorchRouting', 'PytorchRouting.Helpers', 'PytorchRouting.Examples', 'PytorchRouting.CoreLayers',
              'PytorchRouting.UtilLayers', 'PytorchRouting.DecisionLayers', 'PytorchRouting.DecisionLayers.Others',
              'PytorchRouting.DecisionLayers.ReinforcementLearning', 'PytorchRouting.RewardFunctions',
              'PytorchRouting.RewardFunctions.Final', 'PytorchRouting.RewardFunctions.PerAction'],
    url='https://github.com/cle-ros/RoutingNetworks',
    install_requires=['torch>=0.4', 'numpy>=1.12'],
    license='Apache',
    author='Clemens Rosenbaum',
    author_email='cgbr@cs.umass.edu',
    description='a pytorch-based implementation of "RoutingNetworks"'
)
