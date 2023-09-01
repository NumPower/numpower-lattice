<?php

namespace NumPower\Lattice\Core\Activations;

use NumPower\Lattice\Core\Variable;

interface IActivation
{
    /**
     * @param Variable $inputs
     * @return Variable
     */
    function __invoke(Variable $inputs): Variable;
}
