<?php

namespace NumPower\Lattice\Core\Activations;

use NumPower\Lattice\Core\Variable;

interface IActivation
{
    function __invoke(Variable $inputs): Variable;
}