<?php

namespace NumPower\Lattice\Losses;

use \NDArray as nd;

class BinaryCrossEntropy extends Loss
{

    function calculate(\NDArray $target, \NDArray $output): \NDArray|float
    {
        return - ($target * nd::log($output)) + (1 - $target) * nd::log(1 - $output);
    }
}