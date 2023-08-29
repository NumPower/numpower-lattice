<?php

namespace NumPower\Lattice\Core\Initializers;

interface IInitializer
{
    function __invoke(array $shape, bool $use_gpu = false): \NDArray;
}