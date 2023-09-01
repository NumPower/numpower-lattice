<?php

namespace NumPower\Lattice\Initializers;

use NDArray as nd;
use NumPower\Lattice\Core\Initializers\IInitializer;

class Zeros implements IInitializer
{
    /**
     * @param int[] $shape
     * @param bool $use_gpu
     * @return nd
     */
    public function __invoke(array $shape, bool $use_gpu = false): nd
    {
        $zeros = nd::zeros($shape);
        if ($use_gpu) {
            $zeros = $zeros->gpu();
        }
        return $zeros;
    }
}
