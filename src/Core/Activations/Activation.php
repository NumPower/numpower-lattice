<?php

namespace NumPower\Lattice\Core\Activations;

use NDArray;
use NumPower\Lattice\Core\Operation;
use NumPower\Lattice\Exceptions\ValueErrorException;

abstract class Activation implements IActivation
{
    /**
     * @param NDArray|float|int $grad
     * @param Operation $op
     * @return void
     * @throws ValueErrorException
     */
    public function backward(NDArray|float|int $grad, Operation $op): void
    {
        throw new ValueErrorException("Not implemented.");
    }
}
