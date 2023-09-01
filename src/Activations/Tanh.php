<?php

namespace NumPower\Lattice\Activations;

use NDArray as nd;
use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Variable;

class Tanh implements IActivation
{
    /**
     * @param Variable $inputs
     * @return Variable
     */
    public function __invoke(Variable $inputs): Variable
    {
        return Variable::tanh($inputs);
    }
}
