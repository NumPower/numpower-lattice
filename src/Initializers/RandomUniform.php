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

    function initialize(bool $use_gpu): \NDArray
    {
        $a = nd::uniform($this->shape, low: $this->min, high: $this->max);
        if ($use_gpu) {
            $a = $a->gpu();
        }
        return $a;
    }
}