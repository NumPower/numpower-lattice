<?php
namespace NumPower\Lattice\Models;

use \NDArray as nd;
use NumPower\Lattice\Layers\ILayer;
use NumPower\Lattice\Layers\Input;
use NumPower\Lattice\Optimizers\IOptimizer;

class Stack extends Model
{
    /**
     * @param array $layers
     */
    public function __construct(array $layers = []) {

    }

    /**
     * @param ILayer $layer
     * @return Model
     */
    public function add(ILayer $layer): Model
    {
        $this->layers[] = $layer;
        return $this;
    }

    /**
     * @return Model
     */
    public function pop(): Model
    {
        $this->layers = array_pop($this->layers);
        return $this;
    }

    public function save(): void
    {

    }

    public function fit(\NDArray $X, \NDArray $y, int $epochs = 10): void
    {
        for ($current_epoch = 0; $current_epoch < $epochs; $current_epoch++) {
            print("Epoch ". $current_epoch + 1 . "/$epochs\n");
            $loss = parent::trainStep([$X, $y]);
            print("loss: $loss\n");
        }
    }

    public function predict(\NDArray $x): array
    {
        $predictions = [];
        foreach ($x as $sample) {
            $outputs = $this->forward(nd::reshape($sample, [1, count($sample)]));
            $predictions[] = $outputs[0];
        }
        return $predictions;
    }
}