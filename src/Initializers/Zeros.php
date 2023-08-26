<?php

namespace NumPower\Lattice\Initializers;

use \NDArray as nd;

class Zeros extends Initializer
{
    public function initialize(bool $use_gpu): \NDArray
    {
        $a = nd::zeros($this->shape);
        if ($use_gpu) {
            $a = $a->gpu();
        }
        return $a;
    }
}