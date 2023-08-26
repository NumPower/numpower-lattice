<?php

namespace NumPower\Lattice\Initializers;

use \NDArray as nd;

class Zeros extends Initializer
{
    public function initialize(): \NDArray
    {
        return nd::zeros($this->shape);
    }
}