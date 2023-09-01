<?php

namespace NumPower\Lattice;

use NDArray as nd;
use NumPower\Lattice\Core\Operation;

interface IGrad
{
    /**
     * @param nd|int|float $grad
     * @param Operation $op
     * @return void
     */
    public function backward(nd|int|float $grad, Operation $op): void;
}
