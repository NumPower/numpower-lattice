<?php

namespace NumPower\Lattice\Optimizers;

use NumPower\Lattice\Layers\ILayer;

interface IOptimizer
{
    public function adjust(\NDArray $derivatives, ILayer $layer): void;
}