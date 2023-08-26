<?php

namespace NumPower\Lattice\Losses;

use \NDArray as nd;

class BinaryCrossEntropy extends Loss
{

    function calculate(\NDArray $target, \NDArray $output): \NDArray|float
    {
        $target = nd::clip($target, min: 1e-7, max: 1 - 1e-7);
        $loss = -(nd::dot(nd::transpose($output), nd::log($target)) + nd::dot(nd::transpose((1 - $output)), nd::log(1 - $target)));
        return $loss;
    }
}