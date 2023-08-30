<?php

namespace NumPower\Lattice\Losses;

use NumPower\Lattice\Core\Losses\Loss;
use NumPower\Lattice\Core\Variable;

class CategoricalCrossEntropy extends Loss
{
    /**
     * @param \NDArray $true
     * @param Variable $pred
     * @return Variable
     */
    function __invoke(\NDArray $true, Variable $pred): Variable
    {
        // TODO: Implement __invoke() method.
    }
}