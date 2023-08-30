<?php

namespace NumPower\Lattice\Initializers;

use \NDArray as nd;
use NumPower\Lattice\Core\Initializers\Initializer;

class RandomNormal extends Initializer
{
    /**
     * @var int|mixed
     */
    private int|float $scale;

    /**
     * @var int|float
     */
    private int|float $loc;

    /**
     * @param float $loc
     * @param float $scale
     */
    public function __construct(float $loc = 0, float $scale = 0.05) {
        $this->loc = $loc;
        $this->scale = $scale;
    }

    /**
     * @param array $shape
     * @param bool $use_gpu
     * @return \NDArray
     */
    function __invoke(array $shape, bool $use_gpu = false): \NDArray
    {
        return nd::normal($shape, $this->loc, $this->scale);
    }
}