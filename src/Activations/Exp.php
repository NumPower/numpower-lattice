<?php

namespace NumPower\Lattice\Activations;

use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Tensor;
use NumPower\Lattice\Exceptions\ValueErrorException;

class Exp implements IActivation
{
    /**
     * @param Tensor $inputs
     * @return Tensor
     * @throws ValueErrorException
     */
    public function __invoke(Tensor $inputs): Tensor
    {
        return Tensor::exp($inputs);
    }
}
