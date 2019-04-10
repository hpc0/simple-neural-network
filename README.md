# simple-neural-network
Simple neural network for solving problems of selection.




**Create a neural network instance**

`nn = SNN(topology=[3, 4, 5, 3, 1], n_input = 2)`


**Train the neural network**

```
for i in range(1000):
    nn.train(input=[2.45, 4.67], target=[1], lr=0.3)
    nn.apply_training()
```


**Eval some input**

`output = nn.eval([3.55, 2.73])`
