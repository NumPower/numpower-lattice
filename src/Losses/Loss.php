<?php

namespace NumPower\Lattice\Losses;

abstract class Loss implements ILoss
{
    /**
     * @param \NDArray $target
     * @param \NDArray $output
     * @return \NDArray|float
     */
    public function __invoke(\NDArray $target, \NDArray $output): \NDArray|float {
        return $this->calculate($target, $output);
    }
}