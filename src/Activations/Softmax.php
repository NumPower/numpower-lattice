<?php

namespace NumPower\Lattice\Activations;

use NDArray as nd;
use NumPower\Lattice\Core\Activations\Activation;
use NumPower\Lattice\Core\Operation;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Exceptions\ValueErrorException;
use NumPower\Lattice\IGrad;

class Softmax extends Activation implements IGrad
{
    /**
     * @param Variable $inputs
     * @return Variable
     * @throws ValueErrorException
     */
    public function __invoke(Variable $inputs): Variable
    {
        $exps = Variable::exp($inputs);
        $sum_exps = Variable::sum_axis($exps, 1, true);
        return Variable::divide($exps, $sum_exps);
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
