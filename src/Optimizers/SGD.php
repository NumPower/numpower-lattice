<?php

namespace NumPower\Lattice\Optimizers;

use NumPower\Lattice\Layers\ILayer;

class SGD extends Optimizer
{
    /**
     * @var float
     */
    private float $learningRate;

    /**
     * @param float $learningRate
     */
    public function __construct(float $learningRate=0.1)
    {
        $this->learningRate = $learningRate;
    }

    /**
     * @param ILayer $layer
     * @return void
     */
    public function adjust(\NDArray $derivatives, ILayer $layer): void
    {
        $weights = $layer->getWeights();
        $new_weights = $weights + ($derivatives * $this->learningRate);
        $layer->setWeights($new_weights);
    }
}