<?php

namespace NumPower\Lattice\Initializers;

use NDArray as nd;
use NumPower\Lattice\Core\Initializers\Initializer;
use NumPower\Lattice\Exceptions\ValueErrorException;

class GlorotNormal extends Initializer
{
    /**
     * @param int[] $shape
     * @param bool $use_gpu
     * @return nd
     * @throws ValueErrorException
     */
    public function __invoke(array $shape, bool $use_gpu = false): nd
    {
        if (count($shape) != 2) {
            [$fan_in, $fan_out] = (array_key_exists(0, $shape))? [$shape[0], $shape[0]] : [$shape[1], $shape[1]];
        } else {
            [$fan_in, $fan_out] = [$shape[0], $shape[1]];
        }
        $std_dev = sqrt(2.0 / ($fan_in + $fan_out));
        return nd::normal($shape, 0, $std_dev);
    }
}
