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
     * @param Variable $error
     * @param Model $model
     * @return void
     * @throws \Exception
     */
    public function __invoke(Variable $error, Model $model): void
    {
        $error->backward();
        foreach (array_reverse($model->getLayers(), True) as $idx => $layer) {
            if ($layer->isTrainable()) {
                $w = $layer->getTrainableWeights()[0];
                $wd = $w->getArray() - ($w->diff() * $this->learningRate);
                $w->overwriteArray($wd);
            }
        }
    }
}