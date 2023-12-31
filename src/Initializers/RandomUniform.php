<?php

namespace NumPower\Lattice\Initializers;

use NDArray as nd;
use NumPower\Lattice\Core\Initializers\IInitializer;

class RandomUniform implements IInitializer
{
    /**
     * @var float
     */
    private float $min;

    /**
     * @var float
     */
    private float $max;

    /**
     * @param float $min
     * @param float $max
     */
    public function __construct(float $min = 0.0, float $max = 1.0)
    {
        $this->min = $min;
        $this->max = $max;
    }

    /**
     * @param int[] $shape
     * @param bool $use_gpu
     * @return nd
     */
    public function __invoke(array $shape, bool $use_gpu = false): nd
    {
        $a = nd::uniform($shape, low: $this->min, high: $this->max);
        if ($use_gpu) {
            $a = $a->gpu();
        }
        return $a;
    }
}
