<?php

namespace NumPower\Lattice\Initializers;

use \NDArray as nd;

class RandomUniform extends Initializer
{
    private float $min;
    private float $max;

    public function __construct(float $min = 0.0, float $max = 0.9) {
        $this->min = $min;
        $this->max = $max;
    }

    function initialize(): \NDArray
    {
        return nd::uniform($this->shape, low: $this->min, high: $this->max);
    }
}