<?php

namespace NumPower\Lattice;

use NumPower\Lattice\Core\Operation;

interface IGrad
{
    public function backward(\NDArray|int|float $grad, Operation $op): void;
}