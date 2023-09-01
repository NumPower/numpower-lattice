<?php

namespace NumPower\Lattice\Activations;

use \NDArray as nd;
use NumPower\Lattice\Core\Activations\Activation;
use NumPower\Lattice\Core\Operation;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\IGrad;

class ReLU extends Activation implements IGrad
{
    /**
     * @param Variable $inputs
     * @return Variable
     */
    function __invoke(Variable $inputs): Variable
    {
        $new_var = Variable::fromArray(nd::maximum($inputs->getArray(), 0));
        $new_var->setInputs([$inputs, 0]);
        $new_var->registerOperation("relu", $this);
        return $new_var;
    }

    /**
     * @param $grad
     * @param Operation $op
     * @return void
     * @throws \Exception
     */
    public function backward($grad, Operation $op): void
    {
        $op->getArgs()[0]->backward($grad * (nd::greater($op->getArgs()[0]->getArray(), 0)));
    }
}