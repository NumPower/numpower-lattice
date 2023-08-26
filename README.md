> **UNDER DEVELOPMENT** - This library is incomplete and lacks documentation, support, and stability. We recommend its use only by curious people. The current state does not represent the final state of the product.

# NumPower Lattice

Lattice is a PHP framework for creating high-performance neural networks using the NDArray backend. Because of this, Lattice also supports the training and inference of neural models using the GPU.



### Requirements

- NDArray Extension
- PHP 8.2+

### Optional Requirements

- RubixML - Dataset manipulation, normalization and general ML utilities


## Example Code

```php
<?php
require_once "vendor/autoload.php";

use \NDArray as nd;
use NumPower\Lattice\Layers\Dense;
use NumPower\Lattice\Models\Stack;

$model = new Stack();

$model->add(
    new \NumPower\Lattice\Layers\Input(inputShape: [2])
);

$b = new Dense(50, activation: new \NumPower\Lattice\Activations\Tanh());
$model->add(
    $b
);

$c = new Dense(50, activation: new \NumPower\Lattice\Activations\Tanh());
$model->add(
    $c
);

$d = new Dense(1, activation: new \NumPower\Lattice\Activations\Sigmoid());
$model->add(
    $d
);
$model->build(
    optimizer: new \NumPower\Lattice\Optimizers\SGD(learningRate: 0.2),
    loss: new \NumPower\Lattice\Losses\MeanSquaredError()
);

$X = nd::array([[0.5, 0.5],[0.2, 0.3],[0.1,0.2], [0.4, 0.1]]);
$y = nd::array([[1],[0.5],[0.3],[0.5]]);



$model->fit($X, $y, epochs: 5000);

$predictions = $model->predict($X);


foreach ($predictions as $prediction) {
    print_r($prediction);
}
```





