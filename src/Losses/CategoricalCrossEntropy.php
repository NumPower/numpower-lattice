<?php

namespace NumPower\Lattice\Losses;

use Exception;
use NDArray;
use NDArray as nd;
use NumPower\Lattice\Core\Losses\Loss;
use NumPower\Lattice\Core\Operation;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\IGrad;

class CategoricalCrossEntropy extends Loss implements IGrad
{
    /**
     * @var float
     */
    private float $epsilon;

    /**
     * @param float $epsilon
     */
    public function __construct(float $epsilon = 1e-15)
    {
        $this->epsilon = $epsilon;
    }

    /**
     * @param nd $true
     * @param Variable $pred
     * @return Variable
     */
    public function __invoke(nd $true, Variable $pred): Variable
    {

        $true = Variable::fromArray($true);
        $output = Variable::divide($pred, Variable::sum_axis($pred, axis: 1, keepdim: true));
        $m_ones = Variable::fromArray(-1);
        $out = Variable::clip($output, $this->epsilon, 1 - $this->epsilon);
        $out = Variable::divide(
            Variable::multiply(
                Variable::sum(
                    Variable::multiply($true, Variable::log($out))
                ),
                $m_ones
            ),
            Variable::fromArray(count($true->getArray()))
        );
        $out->setInputs([$true, $pred]);
        $out->registerOperation("cce", $this);
        return $out;
    }

    /**
     * @param nd|int|float $grad
     * @param Operation $op
     * @return void
     * @throws Exception
     */
    public function backward(nd|int|float $grad, Operation $op): void
    {
        $denominator = nd::clip(
            ($op->getArgs()[1]->getArray() * (1 - $op->getArgs()[1]->getArray())),
            $this->epsilon,
            1 - $this->epsilon
        );
        $num_samples = $op->getArgs()[1]->getArray()->shape()[0];
        $result = (1 / $num_samples) * ($op->getArgs()[1]->getArray() - $op->getArgs()[0]->getArray());
        $op->getArgs()[1]->backward(
            $result / $denominator
        );
    }
}
