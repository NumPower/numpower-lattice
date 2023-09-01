<?php

namespace NumPower\Lattice\Activations;

use NDArray as nd;
use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Tensor;

class Tanh implements IActivation
{
    /**
     * @param Tensor $inputs
     * @return Tensor
     */
    public function __invoke(Tensor $inputs): Tensor
    {
        return Tensor::tanh($inputs);
    }
}
