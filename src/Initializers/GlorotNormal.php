<?php

namespace NumPower\Lattice\Initializers;

use NDArray as nd;
use NumPower\Lattice\Core\Initializers\Initializer;

class GlorotNormal extends Initializer
{
    /**
     * @param int[] $shape
     * @param bool $use_gpu
     * @return nd
     */
    public function __invoke(array $shape, bool $use_gpu = false): nd
    {
        [$fan_in, $fan_out] = [$shape[0], $shape[1]];
        $std_dev = sqrt(2.0 / ($fan_in + $fan_out));
        return nd::normal($shape, 0, $std_dev);
    }
}
