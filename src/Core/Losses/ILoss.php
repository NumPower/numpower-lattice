<?php

namespace NumPower\Lattice\Core\Losses;

use NDArray;
use NumPower\Lattice\Core\Tensor;

interface ILoss
{
    /**
* @param NDArray $true
* @param Tensor $pred
* @return Tensor
     */
    public function __invoke(NDArray $true, Tensor $pred): Tensor;
}
