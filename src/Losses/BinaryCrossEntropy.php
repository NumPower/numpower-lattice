<?php

namespace NumPower\Lattice\Losses;

use \NDArray as nd;

class BinaryCrossEntropy extends Loss
{
    /**
     * @param nd $target
     * @param nd $output
     * @return nd|float
     */
    function calculate(\NDArray $target, \NDArray $output): \NDArray|float
    {
        $ones_out = nd::ones($output->shape());
        $ones_target = nd::ones($target->shape());
        if ($target->isGPU()) {
            $ones_out = $ones_out->gpu();
            $ones_target = $ones_target->gpu();
        }
        $target = nd::clip($target, min: 1e-7, max: 1 - 1e-7);
        $loss = nd::negative((nd::dot(nd::transpose($output), nd::log($target)) + nd::dot(nd::transpose(($ones_out - $output)), nd::log($ones_target - $target))));
        return nd::average($loss);
    }
}