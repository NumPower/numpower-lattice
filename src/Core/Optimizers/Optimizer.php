<?php

namespace NumPower\Lattice\Core\Optimizers;

use NumPower\Lattice\Core\Models\IModel;

abstract class Optimizer implements IOptimizer
{
    /**
     * @param IModel $model
     * @return void
     */
    public function build(IModel $model): void
    {
        // TODO: Implement build() method.
    }
}