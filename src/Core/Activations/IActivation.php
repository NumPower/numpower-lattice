<?php

namespace NumPower\Lattice\Core\Activations;

use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\IGrad;

interface IActivation
{
    function __invoke(Variable $inputs): Variable;
}