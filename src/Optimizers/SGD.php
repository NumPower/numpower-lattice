<?php

namespace NumPower\Lattice\Optimizers;

use Exception;
use NDArray as nd;
use NumPower\Lattice\Core\Optimizers\Optimizer;
use NumPower\Lattice\Core\Tensor;
use NumPower\Lattice\Models\Model;

class SGD extends Optimizer
{
    /**
     * @var float
     */
    private float $learningRate;

    /**
     * @param float $lr
     */
    public function __construct(float $lr = 0.01)
    {
        $this->learningRate = $lr;
    }

    /**
     * @param Tensor $error
     * @param Model $model
     * @return void
     * @throws Exception
     */
    public function __invoke(Tensor $error, Model $model): void
    {
        $error->backward();
        foreach (array_reverse($model->getLayers(), true) as $layer) {
            if ($layer->isTrainable()) {
                foreach ($layer->getTrainableWeights() as $w) {
                    $wd = $w->getArray() - ($w->diff() * $this->learningRate);
                    $w->overwriteArray($wd);
                }
            }
        }
    }
}
