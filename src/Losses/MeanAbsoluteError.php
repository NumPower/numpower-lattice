<?php

namespace NumPower\Lattice\Losses;

use \NDArray as nd;
use NumPower\Lattice\Core\Losses\Loss;
use NumPower\Lattice\Core\Variable;

class MeanAbsoluteError extends Loss
{

    /**
     * @param \NDArray $true
     * @param Variable $pred
     * @return Variable
     */
    function __invoke(\NDArray $true, Variable $pred): Variable
    {
        $true = Variable::fromArray($true, requireGrad: True);
        $twos = nd::ones($pred->shape()) * 2;
        if ($true->getArray()->isGPU()) {
            $twos = $twos->gpu();
        }
        $twos = Variable::fromArray($twos);
        return Variable::mean(Variable::abs(Variable::subtract($true, $pred)));
    }
}