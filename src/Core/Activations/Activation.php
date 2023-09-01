<?php

namespace NumPower\Lattice\Core\Activations;

use NumPower\Lattice\Core\Operation;
use NumPower\Lattice\Exceptions\ValueErrorException;

abstract class Activation implements IActivation
{
    /**
     * @param $grad
     * @param Operation $op
     * @return void
     * @throws ValueErrorException
     */
    public function backward($grad, Operation $op): void
    {
        throw new ValueErrorException("Not implemented.");
    }
}