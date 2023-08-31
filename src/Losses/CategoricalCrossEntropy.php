<?php

namespace NumPower\Lattice\Losses;

use \NDArray as nd;
use NumPower\Lattice\Core\Losses\Loss;
use NumPower\Lattice\Core\Variable;

class CategoricalCrossEntropy extends Loss
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
     * @param \NDArray $true
     * @param Variable $pred
     * @return Variable
     */
    function __invoke(\NDArray $true, Variable $pred): Variable
    {
        $true = Variable::fromArray($true);
        $output = Variable::divide($pred, Variable::sum_axis($pred, axis: 1, keepdim: True));
        $m_ones = Variable::fromArray(-1);
        $out = Variable::clip($output, $this->epsilon, 1 - $this->epsilon);
        return Variable::divide(Variable::multiply(Variable::sum(Variable::multiply($true, Variable::log($out))), $m_ones), Variable::fromArray(count($true->getArray())));
    }
}