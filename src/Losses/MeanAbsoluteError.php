<?php

namespace NumPower\Lattice\Losses;

use NDArray as nd;
use NumPower\Lattice\Core\Losses\Loss;
use NumPower\Lattice\Core\Tensor;

class MeanAbsoluteError extends Loss
{
    /**
     * @param nd $true
     * @param Tensor $pred
     * @return Tensor
     */
    function __invoke(nd $true, Tensor $pred): Tensor
    {
        $true = Tensor::fromArray($true);
        return Tensor::mean(Tensor::abs(Tensor::subtract($true, $pred)));
    }
}
