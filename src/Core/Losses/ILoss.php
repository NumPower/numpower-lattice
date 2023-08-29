<?php

namespace NumPower\Lattice\Core\Losses;

use NumPower\Lattice\Core\Variable;

interface ILoss
{
    function __invoke(\NDArray $true, Variable $pred): Variable;
}