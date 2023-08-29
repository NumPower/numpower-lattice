<?php

namespace NumPower\Lattice\Activations;

use \NDArray as nd;
use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Exceptions\ValueErrorException;

class Tanh implements IActivation
{
    /**
     * @param Variable $inputs
     * @return Variable
     * @throws ValueErrorException
     */
    public function __invoke(Variable $inputs): Variable
    {
        return $inputs->tanh();
    }
}