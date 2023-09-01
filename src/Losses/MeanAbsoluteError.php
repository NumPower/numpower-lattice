<?php

namespace NumPower\Lattice\Losses;

use NDArray as nd;
use NumPower\Lattice\Core\Losses\Loss;
use NumPower\Lattice\Core\Variable;

class MeanAbsoluteError extends Loss
{
    /**
     * @param nd $true
     * @param Variable $pred
     * @return Variable
     */
    function __invoke(nd $true, Variable $pred): Variable
    {
        $true = Variable::fromArray($true);
        return Variable::mean(Variable::abs(Variable::subtract($true, $pred)));
    }
}
