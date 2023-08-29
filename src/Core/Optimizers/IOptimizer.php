<?php

namespace NumPower\Lattice\Core\Optimizers;

use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Models\Model;

interface IOptimizer
{
    function __invoke(Variable $outputs, Model $model): void;
}