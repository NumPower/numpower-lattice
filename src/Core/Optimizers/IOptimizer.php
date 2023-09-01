<?php

namespace NumPower\Lattice\Core\Optimizers;

use NumPower\Lattice\Core\Models\IModel;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Models\Model;

interface IOptimizer
{
    /**
     * @param Variable $error
     * @param Model $model
     * @return void
     */
    public function __invoke(Variable $error, Model $model): void;

    /**
     * @param IModel $model
     * @return void
     */
    public function build(IModel $model): void;
}
