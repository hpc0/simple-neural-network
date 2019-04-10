# simple-neural-network

Simple neural network for solving problems of selection.

## Installation

The module can be installed via the [pip](https://pip.pypa.io/en/stable/) package manager.


```bash
$ pip install snn
```

## Usage

```python
from snn import SNN


# Create a neural network instance
nn = SNN(topology=[3, 4, 5, 3, 1], n_input = 2)


# train the neural network
for i in range(1000):
    nn.train(input_list=[2.45, 4.67], target=[1], lr=0.3)
    nn.apply_training()


# eval some input
output = nn.eval([3.55, 2.73])
print(output)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[BSD 2-Clause](https://raw.githubusercontent.com/hpc0/simple-neural-network/master/LICENSE)