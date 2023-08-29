<?php

namespace NumPower\Lattice\Initializers;

use \NDArray as nd;
use NumPower\Lattice\Core\Initializers\IInitializer;

class Zeros implements IInitializer
{
    /**
     * @param array $shape
     * @param bool $use_gpu
     * @return nd
     */
    function __invoke(array $shape, bool $use_gpu = false): \NDArray
    {
        $zeros = nd::zeros($shape);
        if ($use_gpu) {
            $zeros = $zeros->gpu();
        }
        return $zeros;
    }
}