<?php

namespace NumPower\Lattice\Optimizers;

use \NDArray as nd;
use NumPower\Lattice\Core\Layers\ILayer;
use NumPower\Lattice\Core\Optimizers\Optimizer;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Models\Model;

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
     * @param Variable $outputs
     * @param Variable $error
     * @param Model $model
     * @return void
     */
    public function __invoke(Variable $outputs, Variable $error, Model $model): void
    {
        $error->backward();
        foreach (array_reverse($model->getLayers(), True) as $idx => $layer) {
            if ($layer->isTrainable()) {
                $w = $layer->getTrainableWeights()[0];
                $w->overwriteArray($w->getArray() - ($w->diff() * $this->learningRate));
            }
        }
    }
}