<?php

namespace NumPower\Lattice\Core\Initializers;

interface IInitializer
{
    /**
     * @param array $shape
     * @param bool $use_gpu
     * @return \NDArray
     */
    function __invoke(array $shape, bool $use_gpu = false): \NDArray;
}
