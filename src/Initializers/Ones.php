<?php

namespace NumPower\Lattice\Initializers;

use NDArray as nd;
use NumPower\Lattice\Core\Initializers\IInitializer;

class Ones implements IInitializer
{
    /**
     * @param int[] $shape
     * @param bool $use_gpu
     * @return nd
     */
    public function __invoke(array $shape, bool $use_gpu = false): nd
    {
        $ones = nd::ones($shape);
        if ($use_gpu) {
            $ones = $ones->gpu();
        }
        return $ones;
    }
}
