<?php

namespace NumPower\Lattice\Core\Optimizers;

use NumPower\Lattice\Core\Models\IModel;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Models\Model;

interface IOptimizer
{
    function __invoke(Variable $error, Model $model): void;
    function build(IModel $model): void;
}