<?php

namespace NumPower\Lattice\Core\Losses;

use NDArray;
use NumPower\Lattice\Core\Variable;

interface ILoss
{
    /**
* @param NDArray $true
* @param Variable $pred
* @return Variable
     */
    public function __invoke(NDArray $true, Variable $pred): Variable;
}
