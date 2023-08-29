<?php /** @noinspection PhpIncompatibleReturnTypeInspection */

namespace NumPower\Lattice\Activations;

use \NDArray as nd;
use NumPower\Lattice\Core\Activations\IActivation;
use NumPower\Lattice\Core\Variable;

/**
 * The sigmoid activation function
 * 1 / (1 + exp(-x))
 */
class Sigmoid implements IActivation
{
    /**
     * @param Variable $inputs
     * @return Variable
     */
    public function __invoke(Variable $inputs): Variable {
        $ones = nd::ones($inputs->getShape());
        return $inputs->negative()->exp()->add($ones)->denominatorDivide($ones);
    }
}