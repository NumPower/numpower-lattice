<?php

namespace NumPower\Lattice\Initializers;

use \NDArray as nd;

class RandomNormal extends Initializer
{
    private float $mean;
    private float $stddev;

    public function __construct(float $mean = 0.0, float $stddev = 0.05) {
        $this->mean = $mean;
        $this->stddev = $stddev;
    }

    function initialize(bool $use_gpu): \NDArray
    {
        $a = nd::normal($this->shape, $this->mean, $this->stddev);
        if ($use_gpu) {
            $a = $a->gpu();
        }
        return $a;
    }
}