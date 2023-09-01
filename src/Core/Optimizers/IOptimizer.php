<?php

namespace NumPower\Lattice\Core\Optimizers;

use NumPower\Lattice\Core\Models\IModel;
use NumPower\Lattice\Core\Tensor;
use NumPower\Lattice\Models\Model;

interface IOptimizer
{
    /**
     * @param Tensor $error
     * @param Model $model
     * @return void
     */
    public function __invoke(Tensor $error, Model $model): void;

    /**
     * @param IModel $model
     * @return void
     */
    public function build(IModel $model): void;
}
