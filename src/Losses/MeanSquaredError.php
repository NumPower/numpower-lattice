<?php

namespace NumPower\Lattice\Losses;

use \NDArray as nd;
use NumPower\Lattice\Losses\Loss;

class MeanSquaredError extends Loss
{
    /**
     * @param \NDArray $target
     * @param \NDArray $output
     * @return float
     */
    public function calculate(\NDArray $target, \NDArray $output): \NDArray|float
    {
        return nd::average(($target - $output) ** 2);
    }
}