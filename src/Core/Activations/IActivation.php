<?php

namespace NumPower\Lattice\Core\Activations;

use NumPower\Lattice\Core\Tensor;

interface IActivation
{
    /**
     * @param Tensor $inputs
     * @return Tensor
     */
    function __invoke(Tensor $inputs): Tensor;
}
