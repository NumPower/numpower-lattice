<?php

namespace NumPower\Lattice\Core;

use NDArray as nd;

interface IGrad
{
    /**
     * @param nd|int|float $grad
     * @param Operation $op
     * @return void
     */
    public function backward(nd|int|float $grad, Operation $op): void;
}
