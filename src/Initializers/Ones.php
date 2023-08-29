<?php

namespace NumPower\Lattice\Initializers;

use \NDArray as nd;
use NumPower\Lattice\Core\Initializers\IInitializer;

class Ones implements IInitializer
{
    /**
     * @param array $shape
     * @param bool $use_gpu
     * @return \NDArray
     */
    function __invoke(array $shape, bool $use_gpu = false): \NDArray
    {
        $ones = nd::ones($shape);
        if ($use_gpu) {
            $ones = $ones->gpu();
        }
        return $ones;
    }
}