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
     * @param Model $model
     * @return void
     */
    public function __invoke(Variable $outputs, Model $model): void
    {
        foreach (array_reverse($model->getLayers(), True) as $idx => $layer) {
            if ($layer->isTrainable()) {
                $diff = $outputs->partialDiff($idx);
                $lr = nd::zeros($diff->shape());
                $lr->fill($this->learningRate);
                if ($diff->isGPU()) {
                    $lr = $lr->gpu();
                }
                $new_weights = [];
                $t_weights = $layer->getTrainableWeights();
                foreach ($t_weights as $idx_w => $weight) {
                    if (count($diff->shape()) == 1) {
                        $diff = nd::reshape($diff, [1, count($diff)]);
                    }
                    $new_weights[$idx_w] = $weight->getArray() + ($diff * $lr);
                }
                foreach ($t_weights as $idx_w => $weight) {
                    $t_weights[$idx_w] = $new_weights[$idx_w];
                }
            }
        }
    }
}