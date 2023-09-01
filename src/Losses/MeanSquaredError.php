<?php

namespace NumPower\Lattice\Losses;

use NDArray as nd;
use NumPower\Lattice\Core\Losses\Loss;
use NumPower\Lattice\Core\Tensor;

class MeanSquaredError extends Loss
{
    /**
     * @param nd $true
     * @param Tensor $pred
     * @return Tensor
     */
    public function __invoke(nd $true, Tensor $pred): Tensor
    {
        $true = Tensor::fromArray($true);
        $twos = nd::ones($pred->shape()) * 2;
        if ($true->getArray()->isGPU()) {
            $twos = $twos->gpu();
        }
        $twos = Tensor::fromArray($twos);
        return Tensor::mean(Tensor::power(Tensor::subtract($true, $pred), $twos));
    }
}
