<?php

namespace NumPower\Lattice\Optimizers;

use \NDArray as nd;
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
        $lr = nd::zeros($derivatives->shape());
        $lr->fill($this->learningRate);

        if ($derivatives->isGPU()) {
            $lr = $lr->gpu();
        }

        $weights = $layer->getWeights();
        $new_weights = $weights + ($derivatives * $lr);
        $layer->setWeights($new_weights);
    }
}