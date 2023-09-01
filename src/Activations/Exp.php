<?php

namespace NumPower\Lattice\Activations;

use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Exceptions\ValueErrorException;

class Exp implements IActivation
{
    /**
     * @param Variable $inputs
     * @return Variable
     * @throws ValueErrorException
     */
    function __invoke(Variable $inputs): Variable
    {
        return Variable::exp($inputs);
    }
}