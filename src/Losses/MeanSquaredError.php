<?php

namespace NumPower\Lattice\Losses;

use \NDArray as nd;
use NumPower\Lattice\Core\Losses\Loss;

class MeanSquaredError extends Loss
{
    /**
     * @param nd $true
     * @param nd $pred
     * @return float
     */
    public function __invoke(\NDArray $true, \NDArray $pred): float
    {
        $twos = nd::ones($pred->shape()) * 2;
        if ($true->isGPU()) {
            $twos = $twos->gpu();
        }
        return nd::average(($true - $pred) ** $twos);
    }
}