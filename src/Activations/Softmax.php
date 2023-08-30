<?php

namespace NumPower\Lattice\Activations;

use \NDArray as nd;
use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Exceptions\ValueErrorException;

class Softmax implements IActivation
{
    /**
     * @param Variable $inputs
     * @return Variable
     * @throws ValueErrorException
     */
    function __invoke(Variable $inputs): Variable
    {
        $exps = Variable::exp($inputs);
        $sum_exps = Variable::sum_axis($exps, 1, true);
        return Variable::divide($exps, $sum_exps);
    }
}