<?php /** @noinspection PhpIncompatibleReturnTypeInspection */

namespace NumPower\Lattice\Activations;

use \NDArray as nd;
use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Variable;
use NumPower\Lattice\Exceptions\ValueErrorException;

/**
 * The sigmoid activation function
 * 1 / (1 + exp(-x))
 */
class Sigmoid implements IActivation
{
    /**
     * @param Variable $inputs
     * @return Variable
     * @throws ValueErrorException
     */
    public function __invoke(Variable $inputs): Variable {
        $minus_ones = Variable::fromArray(-1);
        $ones = Variable::fromArray(1);
        $rtn = Variable::divide(
            $ones,
            Variable::add(
                Variable::exp(
                    Variable::multiply($inputs, $minus_ones)
                ),
                $ones
            )
        );
        return $rtn;
    }
}