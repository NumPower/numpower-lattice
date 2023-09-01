<?php

namespace NumPower\Lattice\Activations;

use NDArray as nd;
use NumPower\Lattice\Core\Activations\Activation;
use NumPower\Lattice\Core\IGrad;
use NumPower\Lattice\Core\Operation;
use NumPower\Lattice\Core\Tensor;
use NumPower\Lattice\Exceptions\ValueErrorException;

class Softmax extends Activation implements IGrad
{
    /**
     * @param Tensor $inputs
     * @return Tensor
     * @throws ValueErrorException
     */
    public function __invoke(Tensor $inputs): Tensor
    {
        $exps = Tensor::exp($inputs);
        $sum_exps = Tensor::sum_axis($exps, 1, true);
        return Tensor::divide($exps, $sum_exps);
    }

    /**
     * @param $grad
     * @param Operation $op
     * @return void
     */
    public function backward($grad, Operation $op): void
    {
        // TODO: Implement backward() method.
    }
}
